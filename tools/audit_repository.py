"""Static hygiene audit for the public CoDiPO repository.

The audit is intentionally conservative. Passing it does not prove scientific
correctness; it checks common release and evaluation-firewall failures.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import asdict, dataclass
from pathlib import Path


INTERNAL_ABSOLUTE_PATH = re.compile(
    r"(?:/home/(?:science|[^/\s\"']+)/|[A-Za-z]:[\\/](?:Users|SPZ_Paper2)[\\/])",
    re.IGNORECASE,
)
CREDENTIAL = re.compile(
    r"(?:api[_-]?key|access[_-]?token|secret|password)\s*=\s*[\"'][^\"']+[\"']",
    re.IGNORECASE,
)
EXTERNAL_COHORT = re.compile(
    r"\b(?:pitt|pitt_external|pitt_loader|lu_external|lu_loader)\b",
    re.IGNORECASE,
)
SELECTION_OPERATION = re.compile(
    r"\b(?:early_stop|best_epoch|best_checkpoint|select_best|rank_score|"
    r"topk|top_k|argsort|argmax|patience)\b",
    re.IGNORECASE,
)
MULTI_COHORT_SCORE = re.compile(
    r"(?:avg|average|mean|weighted|min|max).{0,80}"
    r"(?:pitt|lu|adress|test)|"
    r"(?:pitt|lu|adress|test).{0,80}(?:avg|average|mean|weighted|min|max)",
    re.IGNORECASE | re.DOTALL,
)
MODEL_SELECTION_TARGET = re.compile(
    r"\b(?:checkpoint|ckpt|epoch|seed|repeat|weight|model_path|model_file)\b",
    re.IGNORECASE,
)
DESTRUCTIVE_CHECKPOINT_OPERATION = re.compile(
    r"(?:os\.remove|\.unlink\(|shutil\.move|rmtree)",
    re.IGNORECASE,
)


@dataclass
class Finding:
    path: str
    severity: str
    rule: str
    detail: str


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def audit_python(path: Path, root: Path) -> list[Finding]:
    findings: list[Finding] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = relative(path, root)

    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        findings.append(
            Finding(rel, "ERROR", "PYTHON_SYNTAX", f"line {exc.lineno}: {exc.msg}")
        )
        return findings

    is_gate_tool = "tools" in path.parts

    if not is_gate_tool and INTERNAL_ABSOLUTE_PATH.search(text):
        findings.append(
            Finding(rel, "ERROR", "INTERNAL_ABSOLUTE_PATH", "parameterize paths")
        )
    if not is_gate_tool and CREDENTIAL.search(text):
        findings.append(
            Finding(rel, "ERROR", "CREDENTIAL_LITERAL", "remove secret-like literal")
        )

    is_training_module = (
        "training" in path.parts
        or path.name.startswith("train_")
        or path.stem.endswith("_training")
    )
    is_reporting_module = "manuscript_analysis" in path.parts
    if is_training_module and EXTERNAL_COHORT.search(text):
        findings.append(
            Finding(
                rel,
                "ERROR",
                "EXTERNAL_COHORT_IN_TRAINING_MODULE",
                "move reported-cohort evaluation to a fixed-checkpoint entry point",
            )
        )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        segment = ast.get_source_segment(text, node) or ""
        if (
            not is_reporting_module
            and
            EXTERNAL_COHORT.search(segment)
            and SELECTION_OPERATION.search(segment)
            and MODEL_SELECTION_TARGET.search(segment)
        ):
            findings.append(
                Finding(
                    rel,
                    "ERROR",
                    "EXTERNAL_SELECTION_IN_FUNCTION",
                    f"function {node.name!r} combines evaluation cohorts and selection",
                )
            )
        if (
            not is_reporting_module
            and
            EXTERNAL_COHORT.search(segment)
            and MULTI_COHORT_SCORE.search(segment)
            and MODEL_SELECTION_TARGET.search(segment)
        ):
            findings.append(
                Finding(
                    rel,
                    "ERROR",
                    "MULTI_COHORT_SELECTION_SCORE",
                    f"function {node.name!r} contains a possible cross-cohort score",
                )
            )

    if not is_gate_tool and DESTRUCTIVE_CHECKPOINT_OPERATION.search(text):
        findings.append(
            Finding(
                rel,
                "WARNING",
                "DESTRUCTIVE_FILE_OPERATION",
                "manual review required; release code should preserve provenance",
            )
        )
    return findings


def audit_text(path: Path, root: Path) -> list[Finding]:
    if path.name == "audit_repository.py":
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = relative(path, root)
    findings: list[Finding] = []
    if INTERNAL_ABSOLUTE_PATH.search(text):
        findings.append(
            Finding(rel, "ERROR", "INTERNAL_ABSOLUTE_PATH", "parameterize paths")
        )
    if CREDENTIAL.search(text):
        findings.append(
            Finding(rel, "ERROR", "CREDENTIAL_LITERAL", "remove secret-like literal")
        )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional CSV report. Relative paths are resolved inside the root.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    findings: list[Finding] = []
    tracked_files: list[str] = []
    ignored_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "_audit",
        "_staging",
        "outputs",
        "checkpoints",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        rel = relative(path, root)
        tracked_files.append(rel)
        if path.suffix.lower() == ".ipynb":
            findings.append(
                Finding(rel, "ERROR", "NOTEBOOK_NOT_ALLOWED", "extract one script path")
            )
        elif path.suffix.lower() == ".py":
            findings.extend(audit_python(path, root))
        elif path.suffix.lower() in {".sh", ".json", ".toml", ".yaml", ".yml"}:
            findings.extend(audit_text(path, root))

    report = args.report
    if report:
        if not report.is_absolute():
            report = root / report
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["path", "severity", "rule", "detail"]
            )
            writer.writeheader()
            writer.writerows(asdict(item) for item in findings)

    errors = [item for item in findings if item.severity == "ERROR"]
    warnings = [item for item in findings if item.severity == "WARNING"]
    for item in findings:
        print(f"{item.severity}: {item.path}: {item.rule}: {item.detail}")
    print(
        f"Audited {len(tracked_files)} files: "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
    )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
