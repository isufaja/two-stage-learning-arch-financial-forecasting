# CTTS Input Data Schema

Each experiment consumes one `<symbol>_up.csv` and one `<symbol>_down.csv`.
Rows must be chronologically ordered and aligned by `date`.

## Required Columns

| Column | Purpose |
| --- | --- |
| `date` | Timestamp used to align UP and DOWN records. |
| `pred` | Binary M1 side signal. |
| `meta_label` or `meta_target` | Whether the active M1 signal is reliable. |
| Configured feature columns | Historical inputs such as `close`. |

## Recommended Columns

| Column | Purpose |
| --- | --- |
| `lab` | Realized UP or DOWN class for consensus evaluation. |
| `prediction` | Numerical M1 forecast. |
| `ground_truth` | Realized numerical target. |
| `pred_proba` | M1 directional confidence when available. |

Provider-specific quantiles may also be configured as features. The repository
includes the paper-era Chronos BTC files used by the example configurations.

## Included BTC Checksums

| File | SHA-256 |
| --- | --- |
| `BTC_up.csv` | `8cc4c4ff3cc335d51ba49bb2dd7e773dcf2aadb1b187bc6b0f007f18d0844cee` |
| `BTC_down.csv` | `64ba061d7bb317d0a2761dc3745ce4a2b1fc50fe892ef3f54d00d44a1306f6c1` |
