# Temporal Validation & Chronological Integrity Report

## 1. Chronological Split Architecture

| Split | Date Range | Duration | Usable Records | Percentage |
| :--- | :--- | :--- | :---: | :---: |
| **Training** | 2021-07-14 to 2024-06-26 | ~3.0 years | 112,065 | 56.2% |
| **Validation** | 2024-07-03 to 2024-12-25 | ~6 months | 18,798 | 9.4% |
| **Untouched Test** | 2025-01-01 to 2026-10-21 | ~1.8 years | 68,685 | 34.4% |

## 2. Verification of Strict Temporal Separation

- **Zero Future Lookahead**: Preprocessing imputers and standardizers were fitted exclusively on the Training set.
- **Model Selection Isolation**: All feature ablation decisions and hyperparameter optimization rounds were executed exclusively against the Validation set.
- **Untouched Test Evaluation**: The test split (`2025-01-01` to `2026-10-21`, 68,685 records) was evaluated exactly once after all training and selection decisions were locked.
