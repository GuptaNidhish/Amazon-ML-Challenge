# Business Entity Resolution Pipeline

## Overview
This package provides a scalable, memory-efficient ML pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. It resolves business identities across three noisy data sources (Source 1 reference, Source 2, and Source 3) partitioned by country (US, India, and France).

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── config.py             # File paths and hyperparameters
│   ├── data_loader.py        # Streaming TSV loaders
│   ├── normalization.py     # Fast string normalization (accents, legal suffixes)
│   ├── blocking.py           # Inverted index candidate generation
│   ├── features.py           # 27 pairwise string & address similarity features
│   ├── model.py              # LightGBM model definition & training
│   ├── thresholding.py       # Macro F0.5 threshold optimizer
│   ├── pipeline.py           # End-to-end execution routines
│   └── inference.py          # Output TSV generation & official validation
├── run.py                    # Top-level entrypoint CLI
├── README.md                 # Reproduction instructions
└── requirements.txt          # Pinned dependencies
```

## Reproduction Instructions

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Generate ML Predictions (Submission 2)
To run the full end-to-end inference pipeline using the trained LightGBM model and generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python3 run.py --test --threshold 0.65
```

### 3. Retrain Model (Optional)
To retrain the LightGBM classifier on training data and optimize the decision threshold:
```bash
python3 run.py --train --sample-size 30000
```

### 4. Validate Submission Files
Run the official competition validator:
```bash
python3 student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
