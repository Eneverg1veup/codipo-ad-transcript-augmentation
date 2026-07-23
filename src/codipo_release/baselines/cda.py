"""Train the corrected CDA baseline with an explicit validation boundary.

CDA uses online random-deletion negatives, so it cannot be routed through the
offline-augmentation classifier trainer. This module reproduces the corrected
five-seed, max-length-320 protocol while accepting only training and validation
data. Reported evaluation cohorts are handled later by the fixed-checkpoint
evaluator.
"""

from __future__ import annotations

import argparse
import csv
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
from codipo_release.downstream.train_classifier import (
    binary_metrics,
    read_table,
    sha256,
)


FINAL_CDA_SEEDS = (42, 52, 62, 72, 82)


@dataclass(frozen=True)
class CDASettings:
    base_model: str
    max_length: int = 320
    dropout: float = 0.1
    n_aug: int = 3
    n_neg: int = 3
    delete_probability: float = 0.3
    margin: float = 0.1
    alpha: float = 0.5
    mu: float = 0.5
    epochs: int = 30
    patience: int = 5
    batch_size: int = 8
    evaluation_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.0
    gradient_clip: float = 1.0
    allow_empty_negative: bool = True
    text_column: str = "text1"
    label_column: str = "label"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def random_delete_words(
    text: str,
    *,
    delete_probability: float,
    rng: random.Random,
    allow_empty: bool,
) -> str:
    """Apply CDA's whitespace-token random deletion."""
    words = str(text).split()
    if not words:
        return str(text)
    kept = [
        word for word in words if rng.random() >= delete_probability
    ]
    if not kept and not allow_empty:
        kept = [rng.choice(words)]
    return " ".join(kept)


def build_training_records(
    frame: pd.DataFrame,
    *,
    settings: CDASettings,
    seed: int,
) -> list[dict[str, Any]]:
    missing = {settings.text_column, settings.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing columns {sorted(missing)}.")
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for source_id, row in frame.reset_index(drop=True).iterrows():
        text = str(row[settings.text_column])
        label = int(row[settings.label_column])
        for _ in range(settings.n_aug):
            records.append(
                {
                    "source_id": int(source_id),
                    "text": text,
                    "label": label,
                    "negative_texts": [
                        random_delete_words(
                            text,
                            delete_probability=settings.delete_probability,
                            rng=rng,
                            allow_empty=settings.allow_empty_negative,
                        )
                        for _ in range(settings.n_neg)
                    ],
                }
            )
    return records


def build_cda_classifier(base_model: str, *, dropout: float) -> Any:
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class CDABinaryClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0, :]
            return self.classifier(self.dropout(pooled)).squeeze(-1)

    return CDABinaryClassifier()


def symmetric_bernoulli_kl(
    logits_first: Any, logits_second: Any, *, epsilon: float = 1e-8
) -> Any:
    import torch

    first = torch.sigmoid(logits_first).clamp(epsilon, 1.0 - epsilon)
    second = torch.sigmoid(logits_second).clamp(epsilon, 1.0 - epsilon)
    first_second = first * torch.log(first / second) + (
        1 - first
    ) * torch.log((1 - first) / (1 - second))
    second_first = second * torch.log(second / first) + (
        1 - second
    ) * torch.log((1 - second) / (1 - first))
    return 0.5 * (first_second + second_first).mean()


def cda_margin_loss(
    original_logits: Any,
    negative_logits: Any,
    *,
    margin: float,
    n_neg: int,
) -> Any:
    """Enforce deleted negatives as more AD-like under AD=1 / HC=0."""
    import torch
    import torch.nn.functional as functional

    batch_size = original_logits.size(0)
    original_probability = torch.sigmoid(original_logits)
    negative_probability = (
        torch.sigmoid(negative_logits).view(batch_size, n_neg).mean(dim=1)
    )
    return functional.relu(
        margin + original_probability - negative_probability
    ).mean()


def build_loader(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any,
    settings: CDASettings,
    training: bool,
    seed: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class RecordDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return records[index]

    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["text"] for item in examples],
            padding=True,
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt",
        )
        batch: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(
                [int(item["label"]) for item in examples],
                dtype=torch.float,
            ),
        }
        if training:
            negatives = [
                text
                for item in examples
                for text in item["negative_texts"][: settings.n_neg]
            ]
            negative_encoding = tokenizer(
                negatives,
                padding=True,
                truncation=True,
                max_length=settings.max_length,
                return_tensors="pt",
            )
            batch["negative_input_ids"] = negative_encoding["input_ids"]
            batch["negative_attention_mask"] = negative_encoding[
                "attention_mask"
            ]
        return batch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        RecordDataset(),
        batch_size=(
            settings.batch_size
            if training
            else settings.evaluation_batch_size
        ),
        shuffle=training,
        collate_fn=collate,
        generator=generator,
    )


def build_validation_records(
    frame: pd.DataFrame, *, settings: CDASettings
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": int(index),
            "text": str(row[settings.text_column]),
            "label": int(row[settings.label_column]),
            "negative_texts": [],
        }
        for index, row in frame.reset_index(drop=True).iterrows()
    ]


def validation_metrics(model: Any, loader: Any, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            labels.extend(batch["labels"].int().tolist())
            predictions.extend(
                (torch.sigmoid(logits) >= 0.5).int().cpu().tolist()
            )
    return binary_metrics(labels, predictions)


def train_one_seed(
    *,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    validation_role: str,
    settings: CDASettings,
    seed: int,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    normalized_role = normalize_role(validation_role)
    if normalized_role not in ALLOWED_VALIDATION_ROLES:
        raise ValueError(f"Unsupported validation role: {validation_role}")
    set_seed(seed)
    seed_dir = output_dir / f"seed_{seed}"
    checkpoint = seed_dir / "best_weights.pt"
    if checkpoint.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint}. Use --overwrite explicitly."
        )
    seed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    training_records = build_training_records(
        training_frame, settings=settings, seed=seed
    )
    validation_records = build_validation_records(
        validation_frame, settings=settings
    )
    training_loader = build_loader(
        training_records,
        tokenizer=tokenizer,
        settings=settings,
        training=True,
        seed=seed,
    )
    validation_loader = build_loader(
        validation_records,
        tokenizer=tokenizer,
        settings=settings,
        training=False,
        seed=seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cda_classifier(
        settings.base_model, dropout=settings.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        eps=1e-8,
    )
    total_steps = len(training_loader) * settings.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * settings.warmup_ratio),
        num_training_steps=total_steps,
    )
    binary_cross_entropy = nn.BCEWithLogitsLoss()

    best_accuracy = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "bce": 0.0, "margin": 0.0, "kl": 0.0}
        steps = 0
        for batch in training_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            negative_input_ids = batch["negative_input_ids"].to(device)
            negative_attention_mask = batch[
                "negative_attention_mask"
            ].to(device)
            optimizer.zero_grad(set_to_none=True)
            first_logits = model(input_ids, attention_mask)
            second_logits = model(input_ids, attention_mask)
            bce = 0.5 * (
                binary_cross_entropy(first_logits, labels)
                + binary_cross_entropy(second_logits, labels)
            )
            kl = symmetric_bernoulli_kl(first_logits, second_logits)
            original_logits = 0.5 * (first_logits + second_logits)
            negative_logits = model(
                negative_input_ids, negative_attention_mask
            )
            margin = cda_margin_loss(
                original_logits,
                negative_logits,
                margin=settings.margin,
                n_neg=settings.n_neg,
            )
            loss = bce + settings.alpha * margin + settings.mu * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            for key, value in (
                ("loss", loss),
                ("bce", bce),
                ("margin", margin),
                ("kl", kl),
            ):
                totals[key] += float(value.detach().cpu())
            steps += 1

        metrics = validation_metrics(model, validation_loader, device)
        improved = metrics["accuracy"] > best_accuracy
        if improved:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                **{
                    f"train_{key}": value / max(steps, 1)
                    for key, value in totals.items()
                },
                **{f"validation_{key}": value for key, value in metrics.items()},
                "validation_role": normalized_role,
                "is_best_checkpoint": improved,
            }
        )
        if epochs_without_improvement >= settings.patience:
            break

    history_path = seed_dir / "history.csv"
    pd.DataFrame(history).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )
    provenance = {
        "seed": seed,
        "model_family": "cda",
        "label_convention": "HC=0, AD=1",
        "margin_direction": (
            "mean p_AD(deleted negatives) >= p_AD(original) + margin"
        ),
        "validation_role": normalized_role,
        "selection_metric": "accuracy",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "epochs_ran": len(history),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "settings": asdict(settings),
    }
    (seed_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one CDA seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--validation-role", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seeds", type=parse_seeds, default=FINAL_CDA_SEEDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = CDASettings(
        base_model=args.base_model,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    training = read_table(
        args.training_data,
        text_column=settings.text_column,
        label_column=settings.label_column,
    )
    validation = read_table(
        args.validation_data,
        text_column=settings.text_column,
        label_column=settings.label_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        train_one_seed(
            training_frame=training,
            validation_frame=validation,
            validation_role=args.validation_role,
            settings=settings,
            seed=seed,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        for seed in args.seeds
    ]
    manifest_path = args.output_dir / "fixed_checkpoint_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "checkpoint_id",
                "checkpoint_path",
                "checkpoint_sha256",
                "method",
                "generation_seed",
                "classifier_seed",
                "model_family",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "checkpoint_id": f"cda_seed_{record['seed']}",
                    "checkpoint_path": record["checkpoint"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "method": "CDA",
                    "generation_seed": record["seed"],
                    "classifier_seed": record["seed"],
                    "model_family": "cda",
                }
            )


if __name__ == "__main__":
    main()
