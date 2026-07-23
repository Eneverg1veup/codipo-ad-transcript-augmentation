# CoDiPO

Research code accompanying:

> **Consistency and diversity preference optimization for transcript
> augmentation in Alzheimer's disease detection**

CoDiPO uses image-evidence coverage and residual source--candidate similarity
as relative supervision for constructing preference pairs. The repository
contains the reviewed workflow for candidate generation, proxy scoring,
preference-pair construction, direct preference optimization, augmentation
generation, downstream classification, fixed-checkpoint evaluation, and the
aggregate analyses reported in the manuscript.

## Repository contents

- `src/codipo_release/`: executable Python package;
- `configs/reported_experiment.json`: parameters locked to the reported study;
- `tests/`: CPU-compatible contract and analysis tests;
- `source_data/`: aggregate, non-transcript data supporting manuscript figures
  and tables;
- `docs/reproduction_workflow.md`: ordered reproduction workflow;
- `docs/environment.md`: software, model, and hardware requirements;
- `docs/external_evaluation_firewall.md`: validation and external-evaluation
  rules;
- `tools/`: release-hygiene audit and SHA-256 manifest builder.

## Data boundary

The repository does not redistribute ADReSS, Pitt, or Lu transcripts, the
Cookie Theft image, participant-level predictions, generated candidate text,
raw embeddings, model weights, or trained checkpoints. These materials are
controlled by their original access conditions or are derived from controlled
data.

The files in `source_data/` contain aggregate, non-transcript results only.
Users must obtain the controlled datasets independently and supply their paths
through command-line arguments.

## Installation

The reviewed GPU environment is defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate codipo-release
python -m pip install -e .
```

For local statistical analysis and testing, use
`requirements-analysis-lock.txt`. See `docs/environment.md` for external model
requirements and the SAM 3 installation boundary.

## Workflow

Every executable module exposes its arguments with `--help`.

```bash
# Candidate generation and proxy scoring
python -m codipo_release.pair_construction.generate_candidates --help
python -m codipo_release.proxy_scoring.build_iu_inventory --help
python -m codipo_release.proxy_scoring.score_candidates --help

# Preference-pair construction and DPO training
python -m codipo_release.pair_construction.build_pairs --help
python -m codipo_release.dpo.train_dpo --help

# Augmentation generation
python -m codipo_release.augmentation_inference.generate_adapter_augmentations --help
python -m codipo_release.augmentation_inference.generate_icl_augmentations --help
python -m codipo_release.baselines.eda --help

# Downstream training and fixed-checkpoint evaluation
python -m codipo_release.downstream.train_classifier --help
python -m codipo_release.baselines.cda --help
python -m codipo_release.downstream.evaluate_fixed_checkpoints --help
```

The downstream training entry points accept one explicitly declared validation
input. Lu and Pitt are loaded only by the fixed-checkpoint evaluation and
subsequent analysis stages. ADReSS validation is reported separately; the
external-cohort average is the unweighted Lu--Pitt mean.

The complete order of operations and statistical commands are documented in
`docs/reproduction_workflow.md`.

## Verification

Run the CPU-compatible tests and public release audit:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python tools/audit_repository.py --root .
```

The tests cover the locked prompt and generation defaults, proxy definitions,
pair construction, DPO settings, EDA and corrected CDA protocols, downstream
checkpoint selection, fixed-checkpoint evaluation, external-cohort
aggregation, and manuscript-analysis terminology.

Passing these checks validates the repository interfaces and release hygiene;
it does not replace access to the controlled datasets, external model weights,
or GPU execution required to reproduce the full experiment.

## Citation

Software citation metadata are provided in `CITATION.cff`. The manuscript
citation can be added after publication.

## License

Licensed under the Apache License 2.0. See `LICENSE`.

## Maintainer

Puzhen Su ([supuzhen@163.com](mailto:supuzhen@163.com))

Repository:
<https://github.com/Eneverg1veup/codipo-ad-transcript-augmentation>
