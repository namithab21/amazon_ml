# Business Entity Resolution Pipeline

This repository contains the complete, production-ready machine learning pipeline for the **Business Entity Resolution Challenge** (Amazon ML Challenge 2026).

## Overview
Given noisy business identity records across three independent sources (`Source 1`, `Source 2`, `Source 3`), our pipeline matches records referring to the same real-world business entity using `Source 1` as the reference deduplicated catalog.

The system is designed to handle:
- Large scale: ~1.73M Source 1 entities evaluated against ~10M Source 2/3 candidates (~17.3 Trillion pair space).
- Severe noise: Typographical errors, word transpositions, missing components, DBA/trade names, URL prefixes.
- Unseen domain shift: Zero-shot generalization to test-only regions (**France**) while maintaining performance on **US** and **India**.
- Precision-heavy optimization: Specifically tuned for the competition metric **Macro $F_{0.5}$** ($\beta = 0.5$).

---

## Directory Structure
```
business_entity_resolution/
├── src/
│   ├── config.py         # Path configurations & hyperparameters
│   ├── preprocess.py     # Multilingual normalization, legal term stripping, blocking keys
│   ├── features.py       # C++/SIMD RapidFuzz similarity feature extraction
│   ├── train.py          # Blocker indexing, hard-negative mining, LightGBM training & F_0.5 tuning
│   └── infer.py          # Country-partitioned streaming candidate generation & inference
├── lgb_matcher.txt       # Trained LightGBM model weights
├── requirements.txt      # Pinned dependencies
└── README.md             # End-to-end reproduction guide
```

---

## Architecture & Methodology

### 1. Preprocessing & Normalization (`src/preprocess.py`)
- **Unicode NFKD normalization**: Strips accents and diacritics (`é`, `à`, `ç` $\to$ `e`, `a`, `c`) across French and Indian records.
- **Prefix & Suffix Legal Sanitization**: Strips legal suffixes across US/Global (`Inc`, `LLC`, `Corp`), India (`Pvt Ltd`, `LLP`), and France (`SARL`, `SCI`, `SAS`, `EURL`).
- **DBA & Trade Name Decomposition**: Handles `d/b/a`, `trading as`, `t/a`, `c/o`, extracting true business root tokens.
- **URL Root Extraction**: Extracts core domain tokens from URLs (e.g. `domain.com` $\to$ `domain`).
- **Address Canonicalization**: Standardizes road and street abbreviations (`rd` $\to$ `road`, `st` $\to$ `street`, `blvd` $\to$ `boulevard`, `r.` $\to$ `rue`, `bd` $\to$ `boulevard`).

### 2. Multi-Pass Hybrid Inverted Blocker
- **Dynamic Country Sharding**: Dynamic per-country partitioning (US, India, France) eliminating cross-country comparisons with zero recall loss.
- **Multi-Index Keys**:
  - `NAME_EXACT`: First 14 characters of space-stripped normalized name.
  - `NAME_PREF`: First 6 characters.
  - `NAME_TOK0` / `NAME_TOK1`: Significant non-stopword tokens.
  - `NAME_SORT2`: Sorted token pair to handle word transpositions.
  - `ADDR_NUM_TOK` & `ADDR_NUM_3P`: Street number + street name token (and 3-char prefix for typo tolerance).
  - `NUM_NAME3` & `ZIP_NAME3`: House/Postal number + Name prefix.
- Yields a **92.8%+ recall ceiling** while reducing candidate space to $\le 12$ candidates per entity.

### 3. Feature Engineering (`src/features.py`)
High-speed AVX2/SIMD-accelerated similarity computation:
- **Name Features**: Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Partial Ratio, Length Difference, First-word match.
- **Address Features**: Token Sort Ratio, Jaccard Index, Common Numbers Count, Exact First Number Match.
- **Interaction & Quality Signals**: Address empty indicators, blocking key hits count, source indicator (`S2` vs `S3`).

### 4. Machine Learning Model (`src/train.py`)
- **Classifier**: Histogram-based Gradient Boosted Decision Tree (LightGBM).
- **Training Set**: True positive matches from ground truth + hardest non-matching candidates from the blocking index.
- **Metric Calibration**: Sweep of classification thresholds $\tau \in [0.40, 0.95]$ evaluated on a 20% holdout split using the exact **Macro $F_{0.5}$** formula. Calibrated optimal threshold: $\tau^* = 0.88 - 0.90$.

---

## Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Reproduction Instructions

### Step 1: Train the Matching Model (Optional if using pre-trained weights)
```bash
python src/train.py
```
This trains LightGBM on positive pairs and mined hard negatives, evaluates on the 20% validation split, finds the optimal $F_{0.5}$ threshold, and saves the model to `lgb_matcher.txt`.

### Step 2: Run End-to-End Inference on Test Data
```bash
python src/infer.py
```
This executes country-partitioned streaming candidate generation and model scoring over `dataset/test`, producing:
- `output/candidate_pairs.tsv` (blocking candidate set)
- `output/matching_results.tsv` (final entity matches scored on the leaderboard)

### Step 3: Validate Outputs
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit code `0` confirms both submission files strictly adhere to all formatting rules and constraints.
