# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Apex Entity Resolvers  
**Team Members:** Krish Raj  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a scalable, high-precision hybrid machine learning solution for large-scale Business Entity Resolution across three independent data sources. Our architecture combines a multi-pass inverted blocking engine with dynamic country sharding to reduce the 17.3-trillion potential pair space down to $\le 12$ candidates per entity, while preserving a $>92.8\%$ recall ceiling. A high-speed Gradient Boosted Decision Tree (LightGBM) trained on SIMD-accelerated string, numerical, and structural features is calibrated against a 20% holdout validation set, achieving an optimal Macro $F_{0.5}$ score of **0.9063** by prioritizing precision and rigorously penalizing false merges on singletons.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis of the 2.2M training entities and 1.73M test entities revealed several critical challenges:
1. **Combinatorial Scale**: Matching 1.73M test Source 1 queries against ~10M Source 2 and Source 3 candidates produces $1.73 \times 10^{13}$ pairwise comparisons, making brute-force methods or end-to-end deep cross-encoders computationally infeasible.
2. **Unseen Geographic Domain Shift**: While the training data spans only `US` and `India`, the test set introduces `France` (259,452 Source 1 records). Hardcoded country logic or English-only legal stopwords (e.g., ignoring "SARL", "SCI", "SAS") would severely degrade generalization.
3. **Strict Country Partitioning**: Across 17,250 verified ground-truth links, **0% crossed country boundaries** (100% intra-country consistency), allowing safe per-country sharding with zero recall loss.
4. **Severe Field Asymmetry & Noise**:
   - Missing addresses: Many true matches feature an empty address string `''` in Source 2 or 3, relying entirely on clean name alignment.
   - Alternate trade names / DBAs: Businesses frequently share exact physical addresses (`85 Wayne Avenue`) under completely distinct legal names (`Dréxkor` vs. `Maure Williams Colombier Inc`).
   - Digital artifacts: Name fields frequently embed URLs (e.g., `maurewilliamscolombier.com`).
   - Multilingual text: Devanagari and Kannada scripts, and French diacritics (`é`, `à`, `ç`).
5. **Metric Asymmetry ($F_{0.5}$)**: The evaluation metric weights precision 2× over recall. Merging non-matching businesses produces severe score drops. Furthermore, 5.58% of entities are true singletons; any spurious prediction on a singleton drops its entity score from 1.0 directly to 0.0.

### 2.2 Solution Strategy
We adopt a decoupled, two-stage **Blocking + Machine Learning Classifier** framework engineered for maximum precision:
- **Approach Type**: Multi-Pass Inverted Blocking + C++/SIMD Feature Engineering + Gradient Boosted Decision Trees (LightGBM) + $F_{0.5}$-Optimized Global Calibrated Thresholding.
- **Core Innovation**: 
  - Dynamic country-partitioned multi-key inverted index incorporating URL decomposition, DBA/trade-name extraction, and multi-lingual accent normalization.
  - Hybrid string-numerical similarity feature vectors computed via SIMD-accelerated C++ kernels (RapidFuzz), achieving sub-millisecond per-pair inference.
  - Conservative probability thresholding ($\tau^* = 0.88 - 0.90$) directly calibrated to maximize the precision-heavy Macro $F_{0.5}$ metric and shield singletons from false positives.

---

## 3. Candidate Generation (Blocking)
To compress the 17.3-trillion pair space into a compact candidate pool without sacrificing recall:
- **Blocking keys used**:
  1. `NAME_EXACT`: First 14 characters of space-stripped normalized business name.
  2. `NAME_PREF`: First 6 characters of normalized name.
  3. `NAME_TOK0` / `NAME_TOK1`: Significant non-stopword tokens (filtering generic terms like `the`, `and`, `de`, `du`).
  4. `NAME_SORT2`: Alphabetically sorted token pair to handle word transpositions (`Apex Nippon` vs `Nippon Apex`).
  5. `ADDR_NUM_TOK`: Building/door number combined with the primary street token (`85_wayne`).
  6. `ADDR_NUM_3P`: Building number combined with the 3-letter street prefix (`85_way`), providing robust tolerance to street typos (`wanye` vs `wayne`).
  7. `NUM_NAME3`: House number combined with the 3-letter name prefix (`85_mau`).
  8. `ZIP_NAME3`: 5-digit US/French zip or 6-digit Indian PIN code combined with name prefix.
- **Candidate pairs generated**: Exactly capped at $\le 12$ candidates per Source 1 entity, producing $\approx 15$ million candidate pairs across the entire test set.
- **How true matches were preserved**: The union of orthogonal name and address keys ensures that entities with missing addresses match on name keys, while entities with distinct DBAs/trade names match on street/number keys. Empirical benchmarking on ground truth demonstrated a **92.85% recall ceiling**.

---

## 4. Matching Model

### Features used:
- **Name Features**:
  - `name_jw`: RapidFuzz Jaro-Winkler similarity (optimal for prefix and typo variations).
  - `name_sort_ratio`: Token sort ratio (robust against token reordering).
  - `name_set_ratio`: Token set ratio (captures subset tokens and partial mentions).
  - `name_partial_ratio`: Substring alignment score.
  - `name_len_diff`: Normalized length difference.
  - `first_word_match`: Binary indicator for identical leading token.
- **Address Features**:
  - `addr_token_sort`: Token sort ratio between cleaned address strings.
  - `addr_jaccard`: Word token intersection over union (Jaccard index).
  - `num_common`: Count of shared numeric tokens (building numbers, suites, postal codes).
  - `num_exact_first`: Binary indicator for exact primary house/door number match.
- **Contextual & Interaction Signals**:
  - `is_addr_empty_s1`, `is_addr_empty_c`: Indicators for missing address components, allowing the tree model to route missing-address pairs appropriately.
  - `blocking_hits`: Total number of matching blocking keys (density of shared keys).
  - `source_is_s3`: Source indicator distinguishing Source 2 from Source 3.

### Model Type:
- **Algorithm**: LightGBM (Histogram-based Gradient Boosted Decision Tree).
- **Training Strategy**: Trained on true positive links from `train_ground_truth.tsv` alongside hard negative samples mined directly by the blocking engine (non-matching pairs sharing high key overlap).
- **Hyperparameters**: `max_depth=6`, `num_leaves=31`, `learning_rate=0.05`, `n_estimators=250`, `feature_fraction=0.85`, `bagging_fraction=0.85`.

### Threshold Selection Method:
- We held out a 20% validation split of Source 1 entities and conducted an exhaustive grid search over classification thresholds $\tau \in [0.40, 0.95]$ evaluating the exact competition metric:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- Singletons (5.58% of entities) were evaluated strictly: predicting an empty set yields 1.0, while any false positive yields 0.0.
- The threshold curve monotonically improved as precision increased, peaking at $\tau^* = 0.90$ with a validation score of **0.9063**.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation Split)**: **0.9063** (90.63%)
- **Validation Progression Across Thresholds**:
  - $\tau = 0.50 \to F_{0.5} = 0.8949$
  - $\tau = 0.70 \to F_{0.5} = 0.9020$
  - $\tau = 0.80 \to F_{0.5} = 0.9057$
  - $\tau = 0.90 \to F_{0.5} = 0.9063$
- **Feature Importance (Top Gain)**:
  1. `addr_jaccard` (1,274,630) — Primary spatial discriminator.
  2. `addr_token_sort` (156,564) — Handles street component transposition.
  3. `is_addr_empty_c` (72,380) — Prevents penalization when candidates lack address fields.
  4. `name_set_ratio` (58,226) & `name_partial_ratio` (56,182) — Captures acronyms and trade suffixes.
  5. `blocking_hits` (35,543) — Multi-key reinforcement.
- **Common False Positives (Wrong Merges)**:
  - Co-located distinct businesses sharing identical retail mall or shopping complex addresses (e.g., "Suite 101, City Center") with generic name tokens. Mitigated by strict `name_set_ratio` thresholds.
- **Common False Negatives (Missed Matches)**:
  - Extreme abbreviation coupled with zero address information (e.g., 2-letter acronym with blank address), where insufficient shared information exists for high-confidence classification.

---

## 6. Conclusion
Our solution provides an end-to-end, reproducible, and computationally efficient Entity Resolution pipeline tailored to the Amazon ML Challenge 2026. By engineering a high-recall multi-pass blocking stage, extracting SIMD-accelerated cross-field features, and optimizing LightGBM explicitly for the precision-heavy Macro $F_{0.5}$ metric, we achieve a competitive validation score of **0.9063** with sub-5-minute inference across 1.73 million test entities.

---

## Appendix

### A. Code Artefacts
All runnable code is organized inside `code/business_entity_resolution/`:
- `src/config.py`: Path definitions, candidate caps, and model thresholds.
- `src/preprocess.py`: Multilingual string canonicalization, legal sanitization, and blocking key generators.
- `src/features.py`: RapidFuzz C++ SIMD feature extraction kernel.
- `src/train.py`: Model training, feature importance analysis, and $F_{0.5}$ grid calibration.
- `src/infer.py`: Streaming country-partitioned inference generator producing `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `lgb_matcher.txt`: Self-contained serialized LightGBM model.
- `requirements.txt`: Pinned dependencies (`polars`, `pandas`, `lightgbm`, `rapidfuzz`, `scikit-learn`).

### B. Additional Results
- **Validation Precision**: 92.4%
- **Validation Recall**: 84.1%
- **Singleton Accuracy**: 98.7%
- **Submission Validation**: Verified with `utils/validate_submission.py` (Exit Code 0: `PASS`).
