"""Generate the two proposal-route candidate pools used for pair construction."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codipo_release.pair_construction.candidate_prompts import (
    build_candidate_prompt,
)


@dataclass(frozen=True)
class GenerationSettings:
    base_model: str
    cookie_image: str
    sources_file: str
    output_jsonl: str
    text_column: str = "text1"
    label_column: str = "label"
    id_column: str = "source_id"
    candidates_per_route: int = 40
    max_trials_multiplier: int = 4
    temperature: float = 1.2
    max_new_tokens: int = 320
    seed: int = 2024


def normalize_for_deduplication(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_generated_text(text: str) -> str:
    value = str(text).strip()
    marker = "ASSISTANT:"
    if marker in value:
        value = value.split(marker, 1)[1].strip()
    return value


def read_sources(settings: GenerationSettings) -> Any:
    import pandas as pd

    path = Path(settings.sources_file)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("Sources must be CSV, XLSX or XLS.")
    required = {settings.text_column, settings.label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if settings.id_column not in frame.columns:
        frame[settings.id_column] = range(len(frame))
    labels = {int(value) for value in frame[settings.label_column].tolist()}
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(labels)}")
    return frame


def load_generator(settings: GenerationSettings) -> tuple[Any, Any, Any]:
    import torch
    from PIL import Image
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    processor = LlavaProcessor.from_pretrained(settings.base_model)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        else torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        settings.base_model,
        device_map="auto",
        torch_dtype=dtype,
    )
    image = Image.open(settings.cookie_image).convert("RGB")
    return processor, model, image


def generate_once(
    processor: Any,
    model: Any,
    image: Any,
    prompt_text: str,
    *,
    temperature: float,
    max_new_tokens: int,
) -> str:
    import torch

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device)
        if not torch.is_floating_point(value)
        else value.to(device=device, dtype=model.dtype)
        for key, value in inputs.items()
    }
    prompt_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
    generated = processor.decode(
        output_ids[0][prompt_length:],
        skip_special_tokens=True,
    )
    return extract_generated_text(generated)


def generate_route_pool(
    processor: Any,
    model: Any,
    image: Any,
    *,
    label: int,
    route: str,
    source_text: str,
    target_count: int,
    max_trials_multiplier: int,
    temperature: float,
    max_new_tokens: int,
) -> list[tuple[int, str]]:
    prompt = build_candidate_prompt(label, route, source_text)
    seen: set[str] = set()
    candidates: list[tuple[int, str]] = []
    max_trials = max(target_count, target_count * max_trials_multiplier)
    for trial in range(1, max_trials + 1):
        if len(candidates) >= target_count:
            break
        candidate = generate_once(
            processor,
            model,
            image,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        ).strip()
        key = normalize_for_deduplication(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append((trial, candidate))
    return candidates


def run(settings: GenerationSettings) -> None:
    import numpy as np
    import torch

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)

    sources = read_sources(settings)
    processor, model, image = load_generator(settings)
    output = Path(settings.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in sources.itertuples(index=False):
            row_data = row._asdict()
            source_id = row_data[settings.id_column]
            source_text = str(row_data[settings.text_column])
            label = int(row_data[settings.label_column])
            for route in ("chosen", "rejected"):
                pool = generate_route_pool(
                    processor,
                    model,
                    image,
                    label=label,
                    route=route,
                    source_text=source_text,
                    target_count=settings.candidates_per_route,
                    max_trials_multiplier=settings.max_trials_multiplier,
                    temperature=settings.temperature,
                    max_new_tokens=settings.max_new_tokens,
                )
                for candidate_index, (trial, candidate_text) in enumerate(
                    pool, start=1
                ):
                    record = {
                        "source_id": source_id,
                        "label": label,
                        "prompt_route": route,
                        "candidate_index": candidate_index,
                        "generation_trial": trial,
                        "candidate_text": candidate_text,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> GenerationSettings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--cookie-image", required=True)
    parser.add_argument("--sources-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default="source_id")
    parser.add_argument("--candidates-per-route", type=int, default=40)
    parser.add_argument("--max-trials-multiplier", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2024)
    return GenerationSettings(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
