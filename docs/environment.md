# Environment specification

## Recommended installation

For the full GPU workflow:

```bash
conda env create -f environment.yml
conda activate codipo-release
```

The equivalent pip-oriented dependency list is
`requirements-gpu.txt`. The PyTorch and torchvision builds target CUDA 11.8.
Users with another supported CUDA runtime should select the corresponding
official PyTorch wheels while keeping the Python package versions fixed.

For local statistical analysis, table/figure generation and repository tests,
`requirements-analysis-lock.txt` records the environment in which all 63 tests
passed on 2026-07-23.

## Evidence level

The following DPO package versions are reported in the manuscript and retained
implementation:

- Transformers 4.56.2;
- TRL 0.27.1;
- PEFT 0.17.1.

TRL 0.27.1 requires Python 3.10 or newer, Transformers 4.56.2 or newer,
Datasets 3.0.0 or newer and Accelerate 1.4.0 or newer. The environment files
therefore pin the reported direct packages and use bounded compatible ranges
for those transitive dependencies.

PyTorch 2.4.1+cu118 and the analysis-library versions are the locally audited
submission environment. They are not presented as a recovered byte-for-byte
snapshot of every historical server package.

## External model and data requirements

The repository does not redistribute:

- `llava-1.5-7b-hf`;
- BERT model weights;
- GTE embedding-model weights;
- CLIP model weights;
- the SAM 3 package or checkpoint;
- the Cookie Theft image;
- ADReSS, Pitt or Lu transcripts;
- trained checkpoints.

Install SAM 3 from the upstream distribution associated with the checkpoint.
The module `codipo_release.proxy_scoring.build_iu_inventory` imports
`sam3.model.sam3_image_processor.Sam3Processor` and
`sam3.model_builder.build_sam3_image_model` at runtime. Model and controlled
data paths must be supplied through command-line arguments and must not be
committed to the repository.

## Verification

After installation:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/audit_repository.py --root .
```

Passing the CPU-compatible tests and static audit verifies package contracts,
analysis logic and release hygiene. It does not imply redistribution rights
for third-party models or controlled datasets.
