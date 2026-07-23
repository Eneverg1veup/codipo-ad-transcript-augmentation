"""Build a deterministic SHA-256 manifest for the release tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


IGNORED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "_audit",
    "_staging",
    "checkpoints",
    "outputs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_rows(root: Path, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path == output
            or any(part in IGNORED_PARTS for part in path.relative_to(root).parts)
        ):
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(root, output.resolve())
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} hashes to {output}")


if __name__ == "__main__":
    main()
