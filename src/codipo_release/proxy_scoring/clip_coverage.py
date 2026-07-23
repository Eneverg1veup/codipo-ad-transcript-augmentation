"""CLIP scoring of frozen Cookie Theft information-unit regions."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from codipo_release.proxy_scoring.core import compute_yz, fuse_clip_scores


@dataclass(frozen=True)
class IURegion:
    name: str
    bbox: tuple[float, float, float, float]


def read_iu_inventory(path: str | Path) -> list[IURegion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("information_units", data.get("ius"))
    if not isinstance(data, list):
        raise ValueError("IU inventory must be a list or contain information_units.")
    regions: list[IURegion] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or len(item.get("bbox", [])) != 4:
            raise ValueError(f"Invalid IU record at index {index}.")
        regions.append(
            IURegion(
                name=str(item.get("name", f"IU_{index}")),
                bbox=tuple(float(value) for value in item["bbox"]),
            )
        )
    if not regions:
        raise ValueError("IU inventory is empty.")
    return regions


def split_text_smartly(text: str, max_words: int = 8) -> list[str]:
    """Split text using the executed notebook's sentence-first rule."""

    tokens = re.findall(r"\w+|[^\w\s]", str(text).strip(), flags=re.UNICODE)
    if not tokens:
        return []

    def is_word(token: str) -> bool:
        return bool(re.fullmatch(r"\w+", token, flags=re.UNICODE))

    output: list[str] = []
    buffer: list[str] = []
    word_count = 0
    period_positions: list[tuple[int, int]] = []
    boundaries = {".", "。", "!", "！", "?", "？", ";", "；"}
    for token in tokens:
        buffer.append(token)
        if is_word(token):
            word_count += 1
        if token in boundaries:
            period_positions.append((len(buffer) - 1, word_count))
        if word_count < max_words:
            continue
        cut_index = next(
            (position for position, count in reversed(period_positions)
             if count <= max_words),
            None,
        )
        if cut_index is None:
            current_count = 0
            for index, buffered_token in enumerate(buffer):
                if is_word(buffered_token):
                    current_count += 1
                if current_count == max_words:
                    cut_index = index
                    break
        if cut_index is None:
            continue
        part = " ".join(buffer[: cut_index + 1]).strip()
        part = re.sub(r"\s+([,.!?;:。！？；：])", r"\1", part)
        if part:
            output.append(part)
        buffer = buffer[cut_index + 1 :]
        word_count = sum(is_word(value) for value in buffer)
        period_positions = []
        running_count = 0
        for index, buffered_token in enumerate(buffer):
            if is_word(buffered_token):
                running_count += 1
            if buffered_token in boundaries:
                period_positions.append((index, running_count))
    if buffer:
        part = " ".join(buffer).strip()
        part = re.sub(r"\s+([,.!?;:。！？；：])", r"\1", part)
        if part:
            output.append(part)
    return output


class ClipIUCoverageScorer:
    """Compute global and best-chunk CLIP similarities for fixed IU crops."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        chunk_weight: float = 0.8,
        threshold: float = 0.19,
        max_words_per_chunk: int = 8,
        crop_batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model.eval()
        self.chunk_weight = float(chunk_weight)
        self.threshold = float(threshold)
        self.max_words_per_chunk = int(max_words_per_chunk)
        self.crop_batch_size = int(crop_batch_size)
        self.model_max_length = getattr(
            self.processor.tokenizer, "model_max_length", 77
        )

    def _text_features(self, texts: Sequence[str]) -> Any:
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length,
        ).to(self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        return self.torch.nn.functional.normalize(features, p=2, dim=-1)

    def _image_features(self, images: Sequence[Any]) -> Any:
        inputs = self.processor(images=list(images), return_tensors="pt").to(
            self.device
        )
        with self.torch.no_grad():
            features = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
        return self.torch.nn.functional.normalize(features, p=2, dim=-1)

    def _crop_regions(self, image: Any, regions: Sequence[IURegion]) -> list[Any]:
        width, height = image.size
        crops = []
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            box = (
                max(0, min(int(round(x0)), width)),
                max(0, min(int(round(y0)), height)),
                max(0, min(int(round(x1)), width)),
                max(0, min(int(round(y1)), height)),
            )
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                raise ValueError(f"IU {region.name!r} has an invalid crop.")
            crops.append(image.crop(box))
        return crops

    def score(
        self,
        image: Any,
        text: str,
        regions: Sequence[IURegion],
    ) -> dict[str, Any]:
        crops = self._crop_regions(image, regions)
        full_features = self._text_features([str(text)])
        chunks = split_text_smartly(text, self.max_words_per_chunk)
        chunk_features = self._text_features(chunks or ["[EMPTY]"])
        global_scores: list[float] = []
        chunk_scores: list[float] = []
        for start in range(0, len(crops), self.crop_batch_size):
            image_features = self._image_features(
                crops[start : start + self.crop_batch_size]
            )
            global_batch = image_features @ full_features.T
            chunk_batch = image_features @ chunk_features.T
            global_scores.extend(global_batch[:, 0].detach().cpu().tolist())
            chunk_scores.extend(
                chunk_batch.max(dim=1).values.detach().cpu().tolist()
            )
        final_scores = fuse_clip_scores(
            global_scores, chunk_scores, chunk_weight=self.chunk_weight
        )
        yz = compute_yz(final_scores, threshold=self.threshold)
        return {
            "Y": yz["Y"],
            "Z": yz["Z"],
            "hit_count": yz["hit_count"],
            "total_count": yz["total_count"],
            "text_chunks": chunks,
            "iu_scores": [
                {
                    "name": region.name,
                    "global_score": float(global_score),
                    "chunk_score": float(chunk_score),
                    "final_score": float(final_score),
                    "hit": bool(hit),
                }
                for region, global_score, chunk_score, final_score, hit in zip(
                    regions,
                    global_scores,
                    chunk_scores,
                    final_scores,
                    yz["hit_mask"],
                )
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--iu-inventory", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device")
    parser.add_argument("--chunk-weight", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.19)
    parser.add_argument("--max-words-per-chunk", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    from PIL import Image

    args = parse_args()
    scorer = ClipIUCoverageScorer(
        args.model,
        device=args.device,
        chunk_weight=args.chunk_weight,
        threshold=args.threshold,
        max_words_per_chunk=args.max_words_per_chunk,
    )
    result = scorer.score(
        Image.open(args.image).convert("RGB"),
        args.text,
        read_iu_inventory(args.iu_inventory),
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
