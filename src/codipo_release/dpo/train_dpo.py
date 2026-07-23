"""Train the CoDiPO LoRA adapter from a frozen preference-pair table.

Source boundary: executed cells 0, 1, 3 and 5 of
``20260201_DPO_train.ipynb``. Commented alternatives and every cell after the
executed ``train_dpo_vlm()`` call are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codipo_release.dpo.alignment_prompts import build_alignment_task


REQUIRED_PAIR_COLUMNS = {"anchor_text", "chosen_text", "rejected_text", "label"}


@dataclass(frozen=True)
class DPOSettings:
    base_model: str
    pairs_file: str
    cookie_image: str
    output_dir: str
    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 200
    beta: float = 0.2
    label_smoothing: float = 0.1
    seed: int = 2024
    bf16: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_completion(text: object) -> str:
    value = str(text).strip()
    value = re.sub(r"^\s*(?:ASSISTANT|assistant)\s*:\s*", "", value)
    value = re.sub(r"^\s*(?:USER|user)\s*:\s*", "", value)
    if not value:
        raise ValueError("A chosen or rejected completion is empty.")
    for marker in ("USER:", "ASSISTANT:"):
        if marker in value:
            raise ValueError(f"Completion contains legacy chat marker {marker!r}.")
    return value


def read_pairs(path: Path) -> Any:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Preference pairs must be CSV, XLSX or XLS.")
    missing = REQUIRED_PAIR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing preference-pair columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The preference-pair table is empty.")
    labels = {int(value) for value in frame["label"].tolist()}
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(labels)}")
    return frame


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_processor_and_model(base_model: str) -> tuple[Any, Any]:
    import torch
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    processor = LlavaProcessor.from_pretrained(base_model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    )
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = (
        model.config.vision_feature_select_strategy
    )
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 1
    return processor, model


def find_language_lora_targets(model: Any) -> list[str]:
    import torch

    suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    targets = sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name
        and name.split(".")[-1] in suffixes
    )
    if not targets:
        raise RuntimeError("No language-model LoRA target modules were found.")
    return targets


def build_preference_dataset(
    pairs_file: Path,
    cookie_image: Path,
    processor: Any,
    *,
    seed: int,
) -> Any:
    from datasets import Dataset, Image as HFImage

    frame = read_pairs(pairs_file)
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        label = int(row.label)
        task_text = build_alignment_task(label, str(row.anchor_text))
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task_text},
                ],
            }
        ]
        chosen_messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": clean_completion(row.chosen_text)}],
            }
        ]
        rejected_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": clean_completion(row.rejected_text)}
                ],
            }
        ]
        records.append(
            {
                "images": str(cookie_image),
                "prompt": processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "chosen": processor.apply_chat_template(
                    chosen_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "rejected": processor.apply_chat_template(
                    rejected_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            }
        )
    random.Random(seed).shuffle(records)
    dataset = Dataset.from_list(records)
    return dataset.cast_column("images", HFImage(decode=True))


def validate_multimodal_sample(processor: Any, model: Any, dataset: Any) -> None:
    sample = dataset[0]
    prompt = sample["prompt"]
    if prompt.count("<image>") != 1:
        raise RuntimeError("The rendered prompt must contain exactly one image token.")
    encoded = processor(
        images=sample["images"],
        text=prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    image_token_index = model.config.image_token_index
    image_tokens = int((encoded["input_ids"] == image_token_index).sum().item())
    if image_tokens == 0:
        raise RuntimeError("No image token was found in the encoded prompt.")
    if "pixel_values" not in encoded:
        raise RuntimeError("The processor did not return pixel values.")


def train(settings: DPOSettings) -> None:
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    pairs_file = Path(settings.pairs_file).resolve()
    cookie_image = Path(settings.cookie_image).resolve()
    output_dir = Path(settings.output_dir).resolve()
    if not pairs_file.is_file():
        raise FileNotFoundError(pairs_file)
    if not cookie_image.is_file():
        raise FileNotFoundError(cookie_image)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(settings.seed)
    processor, model = load_processor_and_model(settings.base_model)
    dataset = build_preference_dataset(
        pairs_file,
        cookie_image,
        processor,
        seed=settings.seed,
    )
    validate_multimodal_sample(processor, model, dataset)
    target_modules = find_language_lora_targets(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.epochs,
        bf16=settings.bf16,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        beta=settings.beta,
        label_smoothing=settings.label_smoothing,
        seed=settings.seed,
        data_seed=settings.seed,
        gradient_checkpointing=False,
        max_length=None,
        max_prompt_length=None,
        max_completion_length=None,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    metadata = {
        **asdict(settings),
        "pairs_file": pairs_file.name,
        "pairs_file_sha256": sha256(pairs_file),
        "cookie_image": cookie_image.name,
        "cookie_image_sha256": sha256(cookie_image),
        "number_of_pairs": len(dataset),
        "target_modules": target_modules,
        "checkpoint_selection": "none; final completed adapter",
    }
    with (output_dir / "training_meta.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def parse_args() -> DPOSettings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--cookie-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    return DPOSettings(
        base_model=args.base_model,
        pairs_file=args.pairs_file,
        cookie_image=args.cookie_image,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        bf16=not args.no_bf16,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    train(parse_args())
