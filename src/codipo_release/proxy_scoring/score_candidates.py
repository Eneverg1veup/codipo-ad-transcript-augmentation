"""Score generated candidates with the locked X, Y and Z definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codipo_release.proxy_scoring.clip_coverage import (
    ClipIUCoverageScorer,
    read_iu_inventory,
)
from codipo_release.proxy_scoring.core import (
    ProxySettings,
    build_class_statistics,
    compute_dynamic_bounds,
    estimate_residual_projection,
    evaluate_numeric_candidate,
    ordinary_cosine,
    residual_similarity,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Sources must be CSV, XLSX or XLS.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def map_prompt_type(record: dict[str, Any]) -> str:
    value = str(record.get("prompt_type", record.get("prompt_route", "")))
    if value in {"chosen", "chosen_prompt"}:
        return "chosen_prompt"
    if value in {"rejected", "rejected_prompt"}:
        return "rejected_prompt"
    raise ValueError(f"Unknown proposal route: {value!r}.")


class GTEEmbedder:
    """Executed ModelScope GTE sentence-embedding interface."""

    def __init__(self, model_name_or_path: str, sequence_length: int = 512):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            Tasks.sentence_embedding,
            model=model_name_or_path,
            sequence_length=sequence_length,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        result = self.pipeline(input={"source_sentence": texts})
        return np.asarray(result["text_embedding"], dtype=float)


def score_candidate_records(
    *,
    source_frame: pd.DataFrame,
    candidate_records: list[dict[str, Any]],
    id_column: str,
    text_column: str,
    label_column: str,
    clip_scorer: ClipIUCoverageScorer,
    image: Any,
    regions: list[Any],
    embedder: GTEEmbedder,
    settings: ProxySettings,
) -> list[dict[str, Any]]:
    required = {text_column, label_column}
    missing = required - set(source_frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    sources = source_frame.copy()
    if id_column not in sources.columns:
        sources[id_column] = range(len(sources))
    labels = sources[label_column].astype(int).to_numpy()
    if not set(labels).issubset({0, 1}):
        raise ValueError("Source labels must be binary 0/1.")
    source_texts = sources[text_column].astype(str).tolist()
    source_embeddings = embedder.encode(source_texts)
    center, direction = estimate_residual_projection(
        source_embeddings[labels == 1],
        source_embeddings[labels == 0],
    )

    source_yz = [
        clip_scorer.score(image, text, regions) for text in source_texts
    ]
    class_statistics = {
        label: build_class_statistics(
            [item["Y"] for item, row_label in zip(source_yz, labels)
             if int(row_label) == label],
            [item["Z"] for item, row_label in zip(source_yz, labels)
             if int(row_label) == label],
            consistency_factor=settings.mad_consistency_factor,
        )
        for label in (0, 1)
    }
    source_lookup: dict[str, dict[str, Any]] = {}
    for order, row in enumerate(sources.to_dict(orient="records")):
        source_id = str(row[id_column])
        yz = source_yz[order]
        label = int(row[label_column])
        source_lookup[source_id] = {
            "source_id": row[id_column],
            "source_text": str(row[text_column]),
            "label": label,
            "embedding": source_embeddings[order],
            "source_y": yz["Y"],
            "source_z": yz["Z"],
            "bounds": compute_dynamic_bounds(
                yz["Y"],
                yz["Z"],
                label,
                class_statistics[label],
                risk_multiplier=settings.risk_band_multiplier,
                safe_multiplier=settings.safe_band_multiplier,
            ),
        }

    output: list[dict[str, Any]] = []
    for record in candidate_records:
        source_key = str(record[id_column])
        if source_key not in source_lookup:
            raise ValueError(f"Candidate references unknown source {source_key!r}.")
        source = source_lookup[source_key]
        candidate_text = str(record["candidate_text"])
        candidate_yz = clip_scorer.score(image, candidate_text, regions)
        candidate_embedding = embedder.encode([candidate_text])[0]
        residual_cosine = residual_similarity(
            source["embedding"], candidate_embedding, center, direction
        )
        numeric = evaluate_numeric_candidate(
            candidate_y=candidate_yz["Y"],
            candidate_z=candidate_yz["Z"],
            residual_cosine=residual_cosine,
            origin_cosine=ordinary_cosine(
                source["embedding"], candidate_embedding
            ),
            bounds=source["bounds"],
            residual_threshold=settings.residual_threshold(source["label"]),
        )
        output.append(
            {
                **record,
                id_column: source["source_id"],
                "label": source["label"],
                "source_text": source["source_text"],
                "prompt_type": map_prompt_type(record),
                "source_y": source["source_y"],
                "source_z": source["source_z"],
                **numeric,
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--iu-inventory", required=True, type=Path)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--gte-model", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--settings-json", required=True, type=Path)
    parser.add_argument("--id-column", default="source_id")
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--device")
    parser.add_argument("--clip-threshold", type=float, default=0.19)
    parser.add_argument("--chunk-weight", type=float, default=0.8)
    parser.add_argument("--risk-band-multiplier", type=float, default=0.4)
    parser.add_argument("--safe-band-multiplier", type=float, default=1.0)
    parser.add_argument("--ad-residual-threshold", type=float, default=0.14)
    parser.add_argument("--hc-residual-threshold", type=float, default=0.16)
    return parser.parse_args()


def main() -> None:
    from PIL import Image

    args = parse_args()
    settings = ProxySettings(
        clip_threshold=args.clip_threshold,
        chunk_weight=args.chunk_weight,
        risk_band_multiplier=args.risk_band_multiplier,
        safe_band_multiplier=args.safe_band_multiplier,
        ad_residual_threshold=args.ad_residual_threshold,
        hc_residual_threshold=args.hc_residual_threshold,
    )
    scorer = ClipIUCoverageScorer(
        args.clip_model,
        device=args.device,
        chunk_weight=settings.chunk_weight,
        threshold=settings.clip_threshold,
        max_words_per_chunk=settings.max_words_per_chunk,
    )
    records = score_candidate_records(
        source_frame=read_table(args.sources),
        candidate_records=read_jsonl(args.candidates),
        id_column=args.id_column,
        text_column=args.text_column,
        label_column=args.label_column,
        clip_scorer=scorer,
        image=Image.open(args.image).convert("RGB"),
        regions=read_iu_inventory(args.iu_inventory),
        embedder=GTEEmbedder(args.gte_model),
        settings=settings,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.settings_json.parent.mkdir(parents=True, exist_ok=True)
    args.settings_json.write_text(
        json.dumps(settings.__dict__, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
