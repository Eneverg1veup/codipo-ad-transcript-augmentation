"""Generate source-conditioned augmentations from one fixed LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codipo_release.augmentation_inference.prompts import build_inference_messages


@dataclass(frozen=True)
class InferenceSettings:
    base_model: str
    adapter_dir: str
    cookie_image: str
    sources_file: str
    output_file: str
    text_column: str = "text1"
    label_column: str = "label"
    id_column: str = "source_id"
    augmentations_per_source: int = 2
    temperature: float = 1.2
    max_new_tokens: int = 320
    seed: int = 2024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sources(settings: InferenceSettings) -> Any:
    import pandas as pd

    path = Path(settings.sources_file)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Sources must be CSV, XLSX or XLS.")
    missing = {settings.text_column, settings.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if settings.id_column not in frame.columns:
        frame[settings.id_column] = range(len(frame))
    labels = {int(value) for value in frame[settings.label_column].tolist()}
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(labels)}")
    return frame


def load_adapter(settings: InferenceSettings) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    processor = LlavaProcessor.from_pretrained(settings.adapter_dir)
    base_model = LlavaForConditionalGeneration.from_pretrained(
        settings.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor.patch_size = base_model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = (
        base_model.config.vision_feature_select_strategy
    )
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 1
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    model = PeftModel.from_pretrained(base_model, settings.adapter_dir)
    model.eval()
    image = Image.open(settings.cookie_image).convert("RGB")
    return processor, model, image


def generate_one(
    processor: Any,
    model: Any,
    image: Any,
    *,
    label: int,
    source_text: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    import torch

    messages = build_inference_messages(label, source_text)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generated_ids = output[0, input_length:]
    generated = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return generated.split("ASSISTANT:")[-1].strip()


def write_output(frame: Any, output_file: Path) -> None:
    if output_file.suffix.lower() == ".csv":
        frame.to_csv(output_file, index=False, encoding="utf-8-sig")
    elif output_file.suffix.lower() == ".xlsx":
        frame.to_excel(output_file, index=False)
    else:
        raise ValueError("Output must be CSV or XLSX.")


def run(settings: InferenceSettings) -> None:
    import numpy as np
    import pandas as pd
    import torch

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)

    sources_path = Path(settings.sources_file).resolve()
    image_path = Path(settings.cookie_image).resolve()
    output_path = Path(settings.output_file).resolve()
    if not sources_path.is_file():
        raise FileNotFoundError(sources_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    sources = read_sources(settings)
    processor, model, image = load_adapter(settings)
    rows: list[dict[str, object]] = []
    for row in sources.itertuples(index=False):
        values = row._asdict()
        source_id = values[settings.id_column]
        source_text = str(values[settings.text_column])
        label = int(values[settings.label_column])
        for generation_index in range(1, settings.augmentations_per_source + 1):
            rows.append(
                {
                    "source_id": source_id,
                    "generation_index": generation_index,
                    "text1": generate_one(
                        processor,
                        model,
                        image,
                        label=label,
                        source_text=source_text,
                        temperature=settings.temperature,
                        max_new_tokens=settings.max_new_tokens,
                    ),
                    "label": label,
                    "generation_seed": settings.seed,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(pd.DataFrame(rows), output_path)
    metadata = {
        **asdict(settings),
        "sources_file": sources_path.name,
        "sources_file_sha256": sha256(sources_path),
        "cookie_image": image_path.name,
        "cookie_image_sha256": sha256(image_path),
        "number_of_generated_rows": len(rows),
        "post_generation_proxy_filtering": False,
    }
    with output_path.with_suffix(output_path.suffix + ".meta.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def parse_args() -> InferenceSettings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--cookie-image", required=True)
    parser.add_argument("--sources-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="source_id")
    parser.add_argument("--augmentations-per-source", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2024)
    return InferenceSettings(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
