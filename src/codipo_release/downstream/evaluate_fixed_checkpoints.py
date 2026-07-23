"""Evaluate a pre-locked checkpoint manifest without selecting or mutating it."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codipo_release.downstream.train_classifier import (
    binary_metrics,
    build_classifier,
    read_table,
    sha256,
)


CHECKPOINT_COLUMNS = {
    "checkpoint_id",
    "checkpoint_path",
    "method",
    "generation_seed",
    "classifier_seed",
}
DATASET_COLUMNS = {
    "dataset",
    "dataset_role",
    "data_path",
    "participant_id_column",
    "text_column",
    "label_column",
}


@dataclass(frozen=True)
class EvaluationSettings:
    base_model: str
    max_length: int = 320
    batch_size: int = 32
    dropout: float = 0.1
    use_cls: bool = True
    strict_load: bool = True


def read_csv_records(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def validate_checkpoint_manifest(rows: list[dict[str, str]]) -> None:
    identifiers = [row["checkpoint_id"].strip() for row in rows]
    paths = [row["checkpoint_path"].strip() for row in rows]
    if any(not value for value in identifiers + paths):
        raise ValueError("Checkpoint identifiers and paths must not be empty.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("checkpoint_id values must be unique.")
    for row in rows:
        path = Path(row["checkpoint_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = row.get("checkpoint_sha256", "").strip().lower()
        if expected_hash and sha256(path).lower() != expected_hash:
            raise ValueError(f"Checkpoint hash mismatch: {path}")


def validate_dataset_manifest(rows: list[dict[str, str]]) -> None:
    names = [row["dataset"].strip() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique.")
    for row in rows:
        if not row["dataset_role"].strip():
            raise ValueError("Every dataset requires an explicit dataset_role.")
        if not Path(row["data_path"]).is_file():
            raise FileNotFoundError(row["data_path"])


def load_state_dict(model: Any, checkpoint_path: Path, *, strict: bool) -> None:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(payload)}")
    if "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict, strict=strict)


def build_evaluation_loader(
    frame: pd.DataFrame,
    *,
    tokenizer: Any,
    participant_id_column: str,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, Dataset

    if participant_id_column not in frame.columns:
        raise ValueError(f"Missing participant ID column {participant_id_column!r}.")

    class EvaluationDataset(Dataset):
        def __len__(self) -> int:
            return len(frame)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = frame.iloc[index]
            encoded = tokenizer(
                str(row[text_column]),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(int(row[label_column]), dtype=torch.long)
            item["participant_id"] = str(row[participant_id_column])
            return item

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [key for key in batch[0] if key != "participant_id"]
        output = {
            key: torch.stack([row[key] for row in batch], dim=0) for key in keys
        }
        output["participant_ids"] = [row["participant_id"] for row in batch]
        return output

    return DataLoader(
        EvaluationDataset(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )


def predict(
    model: Any,
    loader: Any,
    device: Any,
    *,
    model_family: str = "bert_two_class",
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            participant_ids = batch.pop("participant_ids")
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            if model_family == "cda":
                logits_ad = model(
                    inputs["input_ids"], inputs["attention_mask"]
                )
                probabilities_ad = torch.sigmoid(logits_ad)
                predictions = (probabilities_ad >= 0.5).long()
            elif model_family == "bert_two_class":
                logits = model(**inputs)["logits"]
                probabilities = torch.softmax(logits, dim=-1)
                probabilities_ad = probabilities[:, 1]
                predictions = logits.argmax(dim=-1)
            else:
                raise ValueError(f"Unsupported model_family: {model_family}")
            for index, participant_id in enumerate(participant_ids):
                if model_family == "cda":
                    logit_hc = float(-logits_ad[index].cpu())
                    logit_ad = float(logits_ad[index].cpu())
                else:
                    logit_hc = float(logits[index, 0].cpu())
                    logit_ad = float(logits[index, 1].cpu())
                rows.append(
                    {
                        "participant_id": participant_id,
                        "y_true": int(labels[index].cpu()),
                        "y_pred": int(predictions[index].cpu()),
                        "y_prob_ad": float(probabilities_ad[index].cpu()),
                        "logit_hc": logit_hc,
                        "logit_ad": logit_ad,
                    }
                )
    return rows


def evaluate_manifests(
    checkpoint_rows: list[dict[str, str]],
    dataset_rows: list[dict[str, str]],
    settings: EvaluationSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch
    from transformers import BertTokenizer

    validate_checkpoint_manifest(checkpoint_rows)
    validate_dataset_manifest(dataset_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(settings.base_model)
    predictions: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    prepared_datasets = []
    for dataset in dataset_rows:
        path = Path(dataset["data_path"])
        frame = read_table(
            path,
            text_column=dataset["text_column"],
            label_column=dataset["label_column"],
        )
        original = (
            pd.read_csv(path)
            if path.suffix.lower() == ".csv"
            else pd.read_excel(path)
        )
        frame[dataset["participant_id_column"]] = original[
            dataset["participant_id_column"]
        ].astype(str)
        loader = build_evaluation_loader(
            frame,
            tokenizer=tokenizer,
            participant_id_column=dataset["participant_id_column"],
            text_column=dataset["text_column"],
            label_column=dataset["label_column"],
            max_length=settings.max_length,
            batch_size=settings.batch_size,
        )
        prepared_datasets.append((dataset, loader, sha256(path)))

    for checkpoint in checkpoint_rows:
        checkpoint_path = Path(checkpoint["checkpoint_path"])
        model_family = (
            checkpoint.get("model_family", "bert_two_class").strip().lower()
            or "bert_two_class"
        )
        if model_family == "cda":
            from codipo_release.baselines.cda import build_cda_classifier

            model = build_cda_classifier(
                settings.base_model, dropout=settings.dropout
            )
        elif model_family == "bert_two_class":
            model = build_classifier(
                settings.base_model,
                dropout=settings.dropout,
                use_cls=settings.use_cls,
            )
        else:
            raise ValueError(f"Unsupported model_family: {model_family}")
        load_state_dict(model, checkpoint_path, strict=settings.strict_load)
        model.to(device)
        checkpoint_hash = sha256(checkpoint_path)
        for dataset, loader, dataset_hash in prepared_datasets:
            dataset_predictions = predict(
                model, loader, device, model_family=model_family
            )
            prefix = {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "checkpoint_sha256": checkpoint_hash,
                "method": checkpoint["method"],
                "model_family": model_family,
                "generation_seed": int(checkpoint["generation_seed"]),
                "classifier_seed": int(checkpoint["classifier_seed"]),
                "dataset": dataset["dataset"],
                "dataset_role": dataset["dataset_role"],
                "dataset_sha256": dataset_hash,
            }
            predictions.extend({**prefix, **row} for row in dataset_predictions)
            metric_values = binary_metrics(
                [row["y_true"] for row in dataset_predictions],
                [row["y_pred"] for row in dataset_predictions],
            )
            metrics.append(
                {
                    **prefix,
                    "n": len(dataset_predictions),
                    **metric_values,
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "max"], default="cls")
    parser.add_argument("--non-strict-load", action="store_true")
    parser.add_argument("--output-predictions", required=True, type=Path)
    parser.add_argument("--output-metrics", required=True, type=Path)
    parser.add_argument("--output-provenance", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_rows = read_csv_records(
        args.checkpoint_manifest, CHECKPOINT_COLUMNS
    )
    dataset_rows = read_csv_records(args.dataset_manifest, DATASET_COLUMNS)
    settings = EvaluationSettings(
        base_model=args.base_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dropout=args.dropout,
        use_cls=args.pooling == "cls",
        strict_load=not args.non_strict_load,
    )
    prediction_frame, metric_frame = evaluate_manifests(
        checkpoint_rows, dataset_rows, settings
    )
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(
        args.output_predictions, index=False, encoding="utf-8-sig"
    )
    metric_frame.to_csv(args.output_metrics, index=False, encoding="utf-8-sig")
    provenance = {
        "checkpoint_manifest_sha256": sha256(args.checkpoint_manifest),
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "checkpoint_count": len(checkpoint_rows),
        "dataset_count": len(dataset_rows),
        "prediction_rows": len(prediction_frame),
        "metric_rows": len(metric_frame),
        "settings": settings.__dict__,
    }
    args.output_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.output_provenance.write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
