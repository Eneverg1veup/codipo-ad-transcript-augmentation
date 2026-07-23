"""Generate the final EDA baseline training tables.

The final experiment used two generated variants per source and five predefined
augmentation seeds. The retained input artifacts contain 216 generated rows
(108 sources by two variants) and no appended originals. The executed trainer
treated each such artifact as the complete training table.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


FINAL_EDA_SEEDS = (1077, 2024, 3407, 4994, 8888)

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "",
}

SynonymProvider = Callable[[str], list[str]]


@dataclass(frozen=True)
class EDASettings:
    num_aug: int = 2
    alpha_sr: float = 0.1
    alpha_ri: float = 0.1
    alpha_rs: float = 0.1
    p_rd: float = 0.1
    official_cleaning: bool = True
    deduplicate_within_source: bool = False
    include_original: bool = False


def get_only_chars(text: str) -> str:
    """Apply the character cleaning used by the official-style EDA baseline."""
    text = (
        str(text)
        .replace("’", "")
        .replace("'", "")
        .replace("-", " ")
        .replace("\t", " ")
        .replace("\n", " ")
        .lower()
    )
    cleaned = "".join(
        character if character in "qwertyuiopasdfghjklzxcvbnm " else " "
        for character in text
    )
    return re.sub(" +", " ", cleaned).strip()


def clean_sentence(text: str, *, official: bool) -> str:
    if official:
        return get_only_chars(text)
    return re.sub(r"\s+", " ", str(text)).strip()


def wordnet_synonyms(word: str) -> list[str]:
    """Return deterministic WordNet synonyms without downloading at runtime."""
    try:
        from nltk.corpus import wordnet

        synsets = wordnet.synsets(word)
    except ImportError as error:
        raise RuntimeError(
            "EDA requires NLTK. Install the optional 'eda' dependencies."
        ) from error
    except LookupError as error:
        raise RuntimeError(
            "The NLTK WordNet corpus is unavailable. Install it before running "
            "the release pipeline; this program never downloads data implicitly."
        ) from error

    synonyms: set[str] = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            candidate = (
                lemma.name().replace("_", " ").replace("-", " ").lower()
            )
            candidate = "".join(
                character
                for character in candidate
                if character in " qwertyuiopasdfghjklzxcvbnm"
            )
            candidate = re.sub(" +", " ", candidate).strip()
            if candidate:
                synonyms.add(candidate)
    synonyms.discard(word)
    return sorted(synonyms)


def synonym_replacement(
    words: list[str],
    count: int,
    rng: random.Random,
    synonym_provider: SynonymProvider,
) -> list[str]:
    output = words.copy()
    candidates = sorted({word for word in words if word not in STOP_WORDS})
    rng.shuffle(candidates)
    replaced = 0
    for word in candidates:
        synonyms = synonym_provider(word)
        if synonyms:
            replacement = rng.choice(synonyms)
            output = [replacement if item == word else item for item in output]
            replaced += 1
        if replaced >= count:
            break
    return " ".join(output).split(" ")


def random_deletion(
    words: list[str], probability: float, rng: random.Random
) -> list[str]:
    if len(words) == 1:
        return words
    output = [word for word in words if rng.uniform(0, 1) > probability]
    if not output:
        return [words[rng.randint(0, len(words) - 1)]]
    return output


def swap_word(words: list[str], rng: random.Random) -> list[str]:
    if len(words) <= 1:
        return words
    first = rng.randint(0, len(words) - 1)
    second = first
    for _ in range(4):
        second = rng.randint(0, len(words) - 1)
        if second != first:
            break
    if second == first:
        return words
    words[first], words[second] = words[second], words[first]
    return words


def random_swap(
    words: list[str], count: int, rng: random.Random
) -> list[str]:
    output = words.copy()
    for _ in range(count):
        output = swap_word(output, rng)
    return output


def add_word(
    words: list[str],
    rng: random.Random,
    synonym_provider: SynonymProvider,
) -> None:
    synonyms: list[str] = []
    for _ in range(10):
        word = words[rng.randint(0, len(words) - 1)]
        synonyms = synonym_provider(word)
        if synonyms:
            break
    if not synonyms:
        return
    words.insert(rng.randint(0, len(words) - 1), synonyms[0])


def random_insertion(
    words: list[str],
    count: int,
    rng: random.Random,
    synonym_provider: SynonymProvider,
) -> list[str]:
    output = words.copy()
    for _ in range(count):
        add_word(output, rng, synonym_provider)
    return output


def eda(
    sentence: str,
    *,
    settings: EDASettings = EDASettings(),
    seed: int,
    synonym_provider: SynonymProvider = wordnet_synonyms,
) -> list[str]:
    """Return the generated variants and optionally the cleaned original."""
    if settings.num_aug < 1:
        raise ValueError("num_aug must be at least one.")
    rng = random.Random(seed)
    sentence = clean_sentence(sentence, official=settings.official_cleaning)
    words = [word for word in sentence.split(" ") if word]
    if not words:
        return [sentence]

    count_per_technique = int(settings.num_aug / 4) + 1
    candidates: list[str] = []
    if settings.alpha_sr > 0:
        count = max(1, int(settings.alpha_sr * len(words)))
        for _ in range(count_per_technique):
            candidates.append(
                " ".join(
                    synonym_replacement(words, count, rng, synonym_provider)
                )
            )
    if settings.alpha_ri > 0:
        count = max(1, int(settings.alpha_ri * len(words)))
        for _ in range(count_per_technique):
            candidates.append(
                " ".join(random_insertion(words, count, rng, synonym_provider))
            )
    if settings.alpha_rs > 0:
        count = max(1, int(settings.alpha_rs * len(words)))
        for _ in range(count_per_technique):
            candidates.append(" ".join(random_swap(words, count, rng)))
    if settings.p_rd > 0:
        for _ in range(count_per_technique):
            candidates.append(
                " ".join(random_deletion(words, settings.p_rd, rng))
            )

    candidates = [
        clean_sentence(item, official=settings.official_cleaning)
        for item in candidates
    ]
    rng.shuffle(candidates)
    output = candidates[: settings.num_aug]
    if settings.include_original:
        output.append(sentence)
    return output


def build_training_table(
    frame: pd.DataFrame,
    *,
    seed: int,
    settings: EDASettings = EDASettings(),
    text_column: str = "text1",
    label_column: str = "label",
    synonym_provider: SynonymProvider = wordnet_synonyms,
) -> pd.DataFrame:
    """Build the EDA table consumed as the complete training input."""
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Input table is missing columns {sorted(missing)}.")
    rows: list[dict[str, object]] = []
    for source_id, source in frame.reset_index(drop=True).iterrows():
        local_seed = int(seed + source_id * 1009 + settings.num_aug * 9176)
        variants = eda(
            str(source[text_column]),
            settings=settings,
            seed=local_seed,
            synonym_provider=synonym_provider,
        )
        if settings.deduplicate_within_source:
            variants = list(dict.fromkeys(variants))
        for index, text in enumerate(variants):
            rows.append(
                {
                    "source_id": int(source_id),
                    "text1": text,
                    "label": int(source[label_column]),
                    "is_original": int(
                        settings.include_original
                        and index == len(variants) - 1
                    ),
                    "aug_index": index,
                    "num_aug": settings.num_aug,
                    "eda_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input table: {path.suffix}")


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one EDA seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="text1")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--seeds", type=parse_seeds, default=FINAL_EDA_SEEDS)
    parser.add_argument("--num-aug", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = EDASettings(num_aug=args.num_aug)
    source = read_table(args.training_data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    for seed in args.seeds:
        table = build_training_table(
            source,
            seed=seed,
            settings=settings,
            text_column=args.text_column,
            label_column=args.label_column,
        )
        output = args.output_dir / (
            f"train_eda_numaug{settings.num_aug}_seed{seed}.csv"
        )
        table.to_csv(output, index=False, encoding="utf-8-sig")
        outputs.append({"seed": seed, "rows": len(table), "path": str(output)})
    manifest = {
        "settings": asdict(settings),
        "output_contract": "full_training_table",
        "outputs": outputs,
    }
    (args.output_dir / "eda_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
