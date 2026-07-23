# Reproduction workflow

## Installation

Create the reviewed environment, then install the package:

```bash
conda env create -f environment.yml
conda activate codipo-release
python -m pip install -e .
```

See `docs/environment.md` for the GPU stack, the locally tested analysis lock,
external model requirements and the SAM 3 installation boundary.

## Execution boundary

GPU/server stages:

1. candidate generation;
2. SAM/IU, CLIP Y/Z and residual-X scoring;
3. preference-pair construction and validation;
4. DPO training;
5. adapter or ICL augmentation generation;
6. downstream BERT or CDA training;
7. fixed-checkpoint inference on ADReSS validation, Lu and Pitt.

CPU/local stages:

1. prediction-lock integrity checks;
2. Lu--Pitt external-cohort aggregation and bootstrap;
3. Tables 2--3 and Supplementary tables;
4. manuscript figures and aggregate source-data export;
5. repository tests and static release audit.

Every module exposes its complete arguments with `python -m MODULE --help`.

## Training and evaluation firewall

The downstream trainer receives training data and one explicitly named
validation dataset. It does not load Lu or Pitt. After training, the
fixed-checkpoint evaluator consumes a manifest of already selected
checkpoints, preserves every row, and evaluates the three reported cohorts.
The evaluator contains no checkpoint ranking, Top-k retention, move or delete
operation.

Representative entry points:

```bash
python -m codipo_release.downstream.train_classifier --help
python -m codipo_release.baselines.cda --help
python -m codipo_release.downstream.evaluate_fixed_checkpoints --help
```

The server command should use explicit absolute paths for controlled inputs and
outputs. Paths must not be written back into release artifacts. Lu and Pitt may
appear only in the fixed-evaluation command, never in a training command.

## Preference workflow

```bash
python -m codipo_release.pair_construction.generate_candidates --help
python -m codipo_release.proxy_scoring.score_candidates --help
python -m codipo_release.pair_construction.build_pairs --help
python -m codipo_release.dpo.train_dpo --help
python -m codipo_release.augmentation_inference.generate_adapter_augmentations --help
```

The final aligned pair artifact must pass the locked checks: 108 sources, 646
pairs, five or six pairs per source, shared conditioning context within each
pair, and exact reconstruction of the recorded pair margin.

## Baselines

```bash
python -m codipo_release.augmentation_inference.generate_icl_augmentations --help
python -m codipo_release.baselines.eda --help
python -m codipo_release.baselines.cda --help
```

EDA outputs two generated rows per source and is passed to the downstream
trainer as the complete training table, matching the five retained 216-row
artifacts. CDA performs online random deletion and therefore has its own
training entry point.

## Statistical lock

The input is a fixed prediction CSV produced only after all checkpoints are
locked:

```bash
python -m codipo_release.manuscript_analysis.statistical_lock \
  --prediction-lock PATH/participant_predictions.csv \
  --output-dir PATH/statistics \
  --n-bootstrap 10000 \
  --bootstrap-seed 20260621
```

ADReSS validation is reported separately. The external-cohort average is the
unweighted mean of Lu and Pitt within each seed group or bootstrap replicate.

Tables are generated from that statistical lock:

```bash
python -m codipo_release.manuscript_analysis.build_tables \
  --statistics-dir PATH/statistics \
  --output-dir PATH/tables \
  --n-bootstrap 10000
```

Figure entry points are `build_figure2`, `build_figure4`, `build_figure5`,
`build_figure6` and `build_figure7` in the same package. Supplementary outputs
use `build_supplementary_tables` and `build_supplementary_k_scaling`. Their
required controlled or aggregate sources are explicit CLI arguments.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python tools/audit_repository.py --root .
```

The audit ignores caches, checkpoints and outputs. Passing the audit means that
the public tree contains no recognized internal absolute path, credential
literal, external-cohort training dependency, destructive checkpoint operation
or cross-cohort model-selection path. It does not replace GPU execution tests
or scientific review.

## Controlled-data boundary

The repository does not contain ADReSS, Pitt or Lu transcript text, raw
participant predictions, model weights, candidate text, source text or raw
embeddings. `source_data/` contains only aggregate non-transcript outputs.
Access-controlled inputs must be obtained under their original data-use terms.
