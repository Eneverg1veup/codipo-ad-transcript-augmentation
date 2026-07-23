"""Train the five-by-five downstream BERT classifier runs.

The training contract accepts one source-training table, one augmentation table
and one explicitly declared validation table per run. Reported evaluation
cohorts are intentionally absent from this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from codipo_release.downstream.checkpoint_selection import (
    ALLOWED_VALIDATION_ROLES,
    normalize_role,
)


REQUIRED_MANIFEST_COLUMNS = {
    "generation_seed",
    "classifier_seed",
    "source_training_data",
    "augmentation_data",
    "validation_data",
    "validation_role",
    "checkpoint_output",
}

ALLOWED_TRAINING_MODES = {
    "source_plus_augmentation",
    "full_training_table",
}
ALLOWED_SHUFFLE_SEED_SOURCES = {"generation_seed", "classifier_seed"}


@dataclass(frozen=True)
class TrainingSettings:
    base_model: str
    max_length: int = 320
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 2e-5
    patience: int = 8
    dropout: float = 0.1
    use_cls: bool = True
    text_column: str = "text1"
    label_column: str = "label"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path, *, text_column: str, label_column: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input table: {path.suffix}")
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}.")
    output = frame[[text_column, label_column]].copy()
    output[text_column] = output[text_column].astype(str)
    output[label_column] = output[label_column].astype(int)
    labels = set(output[label_column].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(labels)}.")
    return output


def read_run_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    validate_run_manifest(rows)
    return rows


def validate_run_manifest(
    rows: list[dict[str, str]],
    *,
    expected_generation_seeds: int = 5,
    expected_classifier_seeds: int = 5,
) -> None:
    if not rows:
        raise ValueError("Run manifest is empty.")
    generation_seeds = {int(row["generation_seed"]) for row in rows}
    classifier_seeds = {int(row["classifier_seed"]) for row in rows}
    expected_rows = expected_generation_seeds * expected_classifier_seeds
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predefined runs, found {len(rows)}.")
    if len(generation_seeds) != expected_generation_seeds:
        raise ValueError(
            f"Expected {expected_generation_seeds} generation seeds, found "
            f"{len(generation_seeds)}."
        )
    if len(classifier_seeds) != expected_classifier_seeds:
        raise ValueError(
            f"Expected {expected_classifier_seeds} classifier seeds, found "
            f"{len(classifier_seeds)}."
        )
    observed = {
        (int(row["generation_seed"]), int(row["classifier_seed"])) for row in rows
    }
    expected = {
        (generation_seed, classifier_seed)
        for generation_seed in generation_seeds
        for classifier_seed in classifier_seeds
    }
    if observed != expected:
        raise ValueError("Manifest must contain the complete predefined 5-by-5 grid.")
    outputs = [str(Path(row["checkpoint_output"])) for row in rows]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Each run requires a unique checkpoint output.")
    roles = {normalize_role(row["validation_role"]) for row in rows}
    if not roles.issubset(ALLOWED_VALIDATION_ROLES):
        raise ValueError(
            f"Unsupported validation roles {sorted(roles - ALLOWED_VALIDATION_ROLES)}."
        )
    modes = {
        row.get("training_mode", "source_plus_augmentation").strip().lower()
        for row in rows
    }
    unsupported_modes = modes - ALLOWED_TRAINING_MODES
    if unsupported_modes:
        raise ValueError(
            f"Unsupported training modes {sorted(unsupported_modes)}."
        )
    shuffle_sources = {
        row.get("shuffle_seed_source", "generation_seed").strip().lower()
        for row in rows
    }
    unsupported_sources = shuffle_sources - ALLOWED_SHUFFLE_SEED_SOURCES
    if unsupported_sources:
        raise ValueError(
            "Unsupported shuffle seed sources "
            f"{sorted(unsupported_sources)}."
        )


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_classifier(base_model: str, *, dropout: float, use_cls: bool) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import BertModel

    class BertBinaryClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bert = BertModel.from_pretrained(base_model)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any = None,
            token_type_ids: Any = None,
            labels: Any = None,
        ) -> dict[str, Any]:
            output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden = output.last_hidden_state
            pooled = hidden[:, 0, :] if use_cls else hidden[:, 1:-1, :].max(dim=1).values
            logits = self.classifier(self.dropout(pooled))
            loss = (
                functional.cross_entropy(logits, labels)
                if labels is not None
                else None
            )
            return {"loss": loss, "logits": logits}

    return BertBinaryClassifier()


def build_loader(
    frame: pd.DataFrame,
    *,
    tokenizer: Any,
    text_column: str,
    label_column: str,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class TextDataset(Dataset):
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
            item["labels"] = torch.tensor(
                int(row[label_column]), dtype=torch.long
            )
            return item

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TextDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def binary_metrics(true: list[int], predicted: list[int]) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
    }


def evaluate(model: Any, loader: Any, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    true: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs)["logits"]
            true.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
    return binary_metrics(true, predicted)


def train_one_run(
    row: dict[str, str],
    settings: TrainingSettings,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from transformers import BertTokenizer

    classifier_seed = int(row["classifier_seed"])
    generation_seed = int(row["generation_seed"])
    set_seed(classifier_seed)
    source_path = Path(row["source_training_data"])
    augmentation_path = Path(row["augmentation_data"])
    validation_path = Path(row["validation_data"])
    checkpoint_path = Path(row["checkpoint_output"])
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. Use --overwrite explicitly."
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    source = read_table(
        source_path,
        text_column=settings.text_column,
        label_column=settings.label_column,
    )
    augmentation = read_table(
        augmentation_path,
        text_column=settings.text_column,
        label_column=settings.label_column,
    )
    validation = read_table(
        validation_path,
        text_column=settings.text_column,
        label_column=settings.label_column,
    )
    training_mode = (
        row.get("training_mode", "source_plus_augmentation").strip().lower()
    )
    if training_mode == "source_plus_augmentation":
        training = pd.concat([source, augmentation], ignore_index=True)
    elif training_mode == "full_training_table":
        training = augmentation.copy()
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")
    shuffle_seed_source = (
        row.get("shuffle_seed_source", "generation_seed").strip().lower()
    )
    shuffle_seed = (
        classifier_seed
        if shuffle_seed_source == "classifier_seed"
        else generation_seed
    )
    training = training.sample(frac=1, random_state=shuffle_seed).reset_index(
        drop=True
    )

    tokenizer = BertTokenizer.from_pretrained(settings.base_model)
    training_loader = build_loader(
        training,
        tokenizer=tokenizer,
        text_column=settings.text_column,
        label_column=settings.label_column,
        max_length=settings.max_length,
        batch_size=settings.batch_size,
        shuffle=True,
        seed=classifier_seed,
    )
    validation_loader = build_loader(
        validation,
        tokenizer=tokenizer,
        text_column=settings.text_column,
        label_column=settings.label_column,
        max_length=settings.max_length,
        batch_size=settings.batch_size,
        shuffle=False,
        seed=classifier_seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        settings.base_model,
        dropout=settings.dropout,
        use_cls=settings.use_cls,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate)

    best_accuracy = -1.0
    best_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        losses = []
        for batch in training_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels)["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                **metrics,
            }
        )
        improved = metrics["accuracy"] > best_accuracy or (
            metrics["accuracy"] == best_accuracy and metrics["f1"] > best_f1
        )
        if improved:
            best_accuracy = metrics["accuracy"]
            best_f1 = metrics["f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
        if bad_epochs >= settings.patience:
            break

    metrics_path = checkpoint_path.with_suffix(".metrics.csv")
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    provenance = {
        "generation_seed": generation_seed,
        "classifier_seed": classifier_seed,
        "validation_role": normalize_role(row["validation_role"]),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_f1": best_f1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_training_sha256": sha256(source_path),
        "augmentation_sha256": sha256(augmentation_path),
        "training_mode": training_mode,
        "shuffle_seed_source": shuffle_seed_source,
        "validation_sha256": sha256(validation_path),
        "settings": asdict(settings),
    }
    checkpoint_path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "max"], default="cls")
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TrainingSettings(
        base_model=args.base_model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        dropout=args.dropout,
        use_cls=args.pooling == "cls",
        text_column=args.text_column,
        label_column=args.label_column,
    )
    rows = read_run_manifest(args.run_manifest)
    results = [
        train_one_run(row, settings, overwrite=args.overwrite) for row in rows
    ]
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
