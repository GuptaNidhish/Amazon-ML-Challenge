# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Solo ML Engineer  
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
We designed a high-precision, country-partitioned entity resolution system specifically optimized for the macro-averaged F₀.₅ metric. By leveraging multi-channel candidate blocking (exact normalized hash, compact string matching, 2-word name indexing, and address composite street-number keys) combined with C++ rapid string similarity features and gradient boosted decision trees (LightGBM), our solution achieves >93.7% precision and 0.7283 macro F₀.₅ on stratified validation while remaining strictly compliant with all model size (≤8B), license, and fair-play constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12.5M train and 11.7M test records revealed three critical characteristics:
1. **Multi-match is the norm:** S1 entities average 3.46 true matches across Source 2 and Source 3; singletons represent only 5.58% of entities.
2. **High name duplication:** Over 38% of S1 business names are shared across different entities, making name similarity alone insufficient and requiring robust address verification.
3. **Noisy name transformations:** Real data contains pervasive abbreviations, domain names (e.g. `teamair.com`), word reordering, legal suffixes, and Indic scripts (Devanagari, Kannada). Address street numbers and street names provide the most resilient disambiguation signal.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Blocking + Pairwise Gradient Boosted Matching + Metric-Aligned Threshold Optimization  
**Core Innovation:** Partitioning by country (US, India, France) combined with composite street-number address blocking and compact domain-stripped name matching, followed by direct grid-search threshold tuning optimizing macro F₀.₅.

---

## 3. Candidate Generation (Blocking)
To overcome the 17-trillion pair O(n²) comparison space:
- **Blocking keys used:**
  1. Exact Normalized Name Hash (accents stripped, legal suffixes removed, punctuation cleaned).
  2. Compact Name String (spaces and domain extensions removed to catch domain/joined names).
  3. Two-word Name Prefix (`first_word + "_" + second_word` to prevent generic single-word explosion).
  4. Address Composite Key (`street_name + "_" + house_number`).
  5. Prefix-4 Name + House Number (`name[:4] + "_" + house_number`).
- **Candidate pairs generated:** Average of 15-35 candidates per S1 entity.
- **Recall preservation:** On sample evaluation, the multi-channel candidate generation captured **97.90%** of all true matches.

---

## 4. Matching Model

**Features used (27 features):**
- **Name features:** Exact match, compact exact match, Levenshtein ratio, partial ratio, token sort ratio, token set ratio, token Jaccard, common token count, first token match, length difference ratio.
- **Address features:** Exact match, ratio, partial ratio, token sort ratio, token set ratio, token Jaccard, numeric token Jaccard, shared numeric count, null address indicator.
- **Cross-field interactions:** Name-address similarity product (`name_sort * addr_sort`), maximum similarity, minimum similarity, high-name/low-address flag, low-name/high-address flag, both-strong flag, Source 3 source indicator.

**Model type:** LightGBM Gradient Boosted Decision Trees (1500 estimators, num_leaves=63, learning_rate=0.05).  
**Threshold selection method:** Grid search directly maximizing macro-averaged F₀.₅ across 0.30 to 0.95 with step 0.05. Optimal threshold found at `tau* = 0.65`.

---

## 5. Results & Error Analysis

- **Macro F₀.₅ Score:** **0.7283** (Validation Set)
  - Micro Precision: **93.71%**
  - Micro Recall: 60.54%
  - Singleton Accuracy: **84.95%**
  - US Macro F₀.₅: 0.7829
  - India Macro F₀.₅: 0.6467
- **Common false positives (wrong merges):** Common brand names or retail chains located within the same commercial plaza sharing street numbers.
- **Common false negatives (missed matches):** Heavily transliterated names without address street numbers in Source 2/3.

---

## 6. Conclusion
By aligning every phase of the pipeline—from multi-channel blocking to LightGBM pairwise scoring and decision threshold tuning—directly with the competition's macro F₀.₅ precision-heavy metric, we built a fast, scalable, and highly accurate solution capable of processing 1.73M test entities in minutes.

---

## Appendix

### A. Code Artefacts
All code is organized under `src/`:
- `src/config.py`: Path definitions and hyperparameters.
- `src/data_loader.py`: High-speed memory-efficient TSV loading.
- `src/normalization.py`: C-speed accent stripping, legal suffix handling, Indic state word mapping.
- `src/evaluation.py`: Exact macro F₀.₅ metric computation.
- `src/blocking.py`: Multi-channel country blocker.
- `src/features.py`: 27 C++ rapidfuzz pairwise features.
- `src/model.py`: LightGBM training and inference.
- `src/thresholding.py`: Direct macro F₀.₅ threshold optimizer.
- `src/inference.py`: Output TSV generation and official validator integration.
- `run.py`: Top-level CLI (`--baseline`, `--train`, `--test`).
