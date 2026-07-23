# External-evaluation firewall

## Dataset roles

- **Training data**: may be split into fold-local training and validation
  partitions.
- **Validation data**: the single dataset explicitly assigned to early stopping
  and checkpoint selection. Its provenance must be stored as either
  `train_internal_validation` or `adress_validation`.
- **ADReSS validation**: the official ADReSS test partition when it is assigned
  a source-domain validation role. If it contributes to model selection, it
  must be reported as validation rather than independent held-out evidence.
- **Lu and de-overlapped Pitt**: external evaluation cohorts.

## Required control flow

1. Fix augmentation seeds, downstream training seeds and folds before training.
2. Declare exactly one validation dataset and record its role.
3. Train and select the epoch/checkpoint using only that validation dataset.
4. Save the selected checkpoint and its SHA-256.
5. Close the training process.
6. In a separate evaluation entry point, load the fixed checkpoint and evaluate
   ADReSS validation, Lu and Pitt.
7. Aggregate all prespecified seeds/folds. Do not select a top subset by
   evaluation-cohort performance.

## Forbidden operations

- an average, weighted average, minimum or maximum across multiple reported
  cohorts used to select an epoch, checkpoint, repeat or seed;
- any use of Lu or Pitt to select an epoch, checkpoint, repeat or seed;
- evaluating an external cohort inside an early-stopping loop;
- retaining only the top-performing external-evaluation runs;
- encoding evaluation metrics in filenames and later using those values to rank
  checkpoints;
- deleting non-selected checkpoints before provenance is closed;
- calling ADReSS validation simply `validation` in code or documentation.

## Release terminology

Use `cv_validation` for a training-internal split and `adress_validation` for
the official ADReSS partition when it is assigned a validation role. Use
`lu_external` and `pitt_external` only for post-selection evaluation.
