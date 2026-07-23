"""Generate final Direct, Rewrite and Imitation ICL controls."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from codipo_release.augmentation_inference.icl_prompts import build_icl_prompt
from codipo_release.downstream.train_classifier import sha256


FINAL_ICL_SEEDS = (5942, 3012, 4951, 2921, 9032)
FINAL_ICL_METHODS = ("direct", "rewrite", "imitation")


@dataclass(frozen=True)
class ICLGenerationSettings:
    samples_per_source: int = 2
    temperature: float = 1.2
    max_new_tokens: int = 320
    methods: tuple[str, ...] = FINAL_ICL_METHODS


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_assistant_prefix(text: str) -> str:
    return str(text).split("ASSISTANT:")[-1].strip()


def read_source(
    path: Path, *, text_column: str, label_column: str
) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported source table: {path.suffix}")
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Source table is missing columns {sorted(missing)}.")
    output = frame[[text_column, label_column]].copy()
    output[text_column] = output[text_column].astype(str)
    output[label_column] = output[label_column].astype(int)
    labels = set(output[label_column].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(labels)}.")
    return output


def load_model_and_processor(base_model: str) -> tuple[Any, Any]:
    import torch
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    processor = LlavaProcessor.from_pretrained(base_model)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    return model, processor


def generate_one_table(
    frame: pd.DataFrame,
    *,
    method: str,
    seed: int,
    image: Any,
    model: Any,
    processor: Any,
    settings: ICLGenerationSettings,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    import torch

    method = method.strip().lower()
    if method not in settings.methods:
        raise ValueError(f"Unsupported ICL method: {method}")
    set_seed(seed)
    rows: list[dict[str, Any]] = []
    for source_id, source in frame.reset_index(drop=True).iterrows():
        original_text = str(source[text_column])
        label = int(source[label_column])
        system_text, user_text = build_icl_prompt(
            method, label, original_text
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        for sample_index in range(settings.samples_per_source):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                    do_sample=True,
                    temperature=settings.temperature,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            generated = processor.decode(
                output[0, input_length:], skip_special_tokens=True
            )
            rows.append(
                {
                    "source_id": int(source_id),
                    "sample_index": sample_index,
                    "text1": strip_assistant_prefix(generated),
                    "label": label,
                    "method": method,
                    "generation_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return values


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(
        item.strip().lower() for item in value.split(",") if item.strip()
    )
    invalid = set(methods) - set(FINAL_ICL_METHODS)
    if not methods or invalid:
        raise argparse.ArgumentTypeError(
            f"Methods must be drawn from {FINAL_ICL_METHODS}; found {sorted(invalid)}."
        )
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seeds", type=parse_ints, default=FINAL_ICL_SEEDS)
    parser.add_argument(
        "--methods", type=parse_methods, default=FINAL_ICL_METHODS
    )
    parser.add_argument("--samples-per-source", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ICLGenerationSettings(
        samples_per_source=args.samples_per_source,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        methods=args.methods,
    )
    frame = read_source(
        args.training_data,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    image = Image.open(args.image).convert("RGB")
    model, processor = load_model_and_processor(args.base_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for method in settings.methods:
        for seed in args.seeds:
            table = generate_one_table(
                frame,
                method=method,
                seed=seed,
                image=image,
                model=model,
                processor=processor,
                settings=settings,
                text_column=args.text_column,
                label_column=args.label_column,
            )
            output_path = args.output_dir / f"icl_{method}_seed_{seed}.csv"
            table.to_csv(output_path, index=False, encoding="utf-8-sig")
            outputs.append(
                {
                    "method": method,
                    "seed": seed,
                    "rows": len(table),
                    "path": str(output_path),
                    "sha256": sha256(output_path),
                }
            )
    manifest = {
        "settings": asdict(settings),
        "training_data_sha256": sha256(args.training_data),
        "image_sha256": sha256(args.image),
        "outputs": outputs,
    }
    (args.output_dir / "icl_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

