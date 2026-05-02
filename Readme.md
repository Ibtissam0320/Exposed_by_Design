# Exposed by Design: Privacy Attacks & Defenses for Knowledge Graphs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4.1-red)](https://pytorch.org/)

This repository contains the code and resources to reproduce the main results on **topology-based privacy attacks** against Knowledge Graphs and their **defense mechanisms**. We demonstrate three distinct attack strategies and three defense techniques, with comprehensive evaluation across multiple KG datasets (NELL, FB15k-237, HealthKG).

## 📋 Overview

### Attack Strategies
- **Attack 1: Link Inference** – Predicts sensitive links using topological features from the public graph
- **Attack 2: Tail Inference** – Infers missing tail entities for sensitive relations
- **Attack 3: Graph Reconstruction** – Reconstructs entire sensitive subgraphs from public topology

### Defense Mechanisms
- **K-Anonymity** – Ensures each node appears in at least k sensitive triples, achieving indistinguishability
- **Randomized Response** – Adds probabilistic noise to degrade attack accuracy while maintaining utility
- **CHAMELEON** – Adaptive defense selecting strategies based on node structural profiles

##  Quick Start

### 1. Installation

**Clone and install dependencies:**
```bash
git clone <repository-url>
cd Exposed_by_Design
pip install -r requirements.txt
```

**Main requirements:**
- PyTorch 2.4.1
- NetworkX 3.1
- Scikit-learn 1.3.2
- Pandas 2.0.3
- NumPy 1.24.4
- CUDA (optional, for GPU acceleration)

### 2. Prepare Datasets

**Download datasets:**
- **NELL**: https://rtw.ml.cmu.edu/rtw/
- **FB15k-237**: https://huggingface.co/datasets/KGraph/FB15k-237
- **HealthKG**: https://github.com/Boreico/KGE_QCB_Project

**Split into public/private portions:**
```bash
python dataprocessing/split.py \
  --global_path /path/to/full_kg.tsv \
  --relation "sensitive_relation_1" \
  --relation "sensitive_relation_2" \
  --outdir /path/to/output/
```

### 3. Run Attacks & Defenses

See [Reproducing Results](#reproducing-results) section below.

## 🖥️ Hardware Requirements

**Recommended:**
- NVIDIA GPU (RTX 3090, A100, or equivalent with 24GB+ VRAM)
- 16+ GB RAM
- CPU: 8+ cores
- Linux (Ubuntu 20.04 or later)

**Minimum (CPU-only, slower):**
- 8GB RAM
- 4+ cores
- Any OS with Python 3.7+

**Note:** Smaller dataset experiments (NELL) can run on CPU in reasonable time (~1-2 hours).

## 📁 Project Structure

```
.
├── attacks/                           # Attack implementations
│   ├── attack1_head.py               # Link inference (head prediction)
│   ├── attack1_tail.py               # Link inference (tail prediction)
│   ├── attack2.py                    # Tail entity inference
│   └── attack3.py                    # Graph reconstruction
│
├── defenses/                          # Defense mechanisms
│   ├── defense_kanonymity.py         # K-Anonymity implementation
│   ├── defense_randomized_response.py # Randomized Response implementation
│   ├── chameleon_defense.py          # CHAMELEON adaptive defense
│   ├── kanon_runner.sh               # K-Anonymity batch runner
│   ├── rr_runner.sh                  # Randomized Response batch runner
│   └── chameleon_runner.sh           # CHAMELEON batch runner
│
├── dataprocessing/
│   ├── split.py                      # KG splitting utility
│   └── readme.md                     # Dataset documentation
│
├── experiments/                       # Evaluation & analysis
│   ├── attack1_featuresstudy.py      # Feature importance for Attack 1
│   ├── attack2_featuresstudy.py      # Feature importance for Attack 2
│   ├── attack3_featuresstudy.py      # Feature importance for Attack 3
│   ├── utility_LinkPrediction.py     # Link prediction utility evaluation
│   ├── features_distribution.py      # Feature distribution analysis
│   ├── runner1.sh                    # Batch runner for Attack 1
│   ├── runner2.sh                    # Batch runner for Attack 2
│   └── runner3.sh                    # Batch runner for Attack 3
│
├── requirements.txt                   # Python dependencies
└── Readme.md                          # This file
```

## 📊 Reproducing Results

### Step 1: Data Preparation

```bash
python dataprocessing/split.py \
  --global_path <full_kg.tsv> \
  --relation <relation_name> \
  --outdir ./prepared_data/
```

### Step 2: Run Attacks

**Option A: Individual attacks**
```bash
# Attack 1 - Head prediction
python attacks/attack1_head.py \
  --public-path ./prepared_data/public.tsv \
  --sens-path ./prepared_data/sensitive.tsv \
  --outdir ./results/attack1_head/ \
  --seed 42

# Attack 1 - Tail prediction
python attacks/attack1_tail.py \
  --public-path ./prepared_data/public.tsv \
  --sens-path ./prepared_data/sensitive.tsv \
  --outdir ./results/attack1_tail/ \
  --seed 42

# Attack 2 - Tail inference (requires Attack 1 scores)
python attacks/attack2.py \
  --public-path ./prepared_data/public.tsv \
  --sens-path ./prepared_data/sensitive.tsv \
  --attack1-head-scores ./results/attack1_head/scores.tsv \
  --attack1-tail-scores ./results/attack1_tail/scores.tsv \
  --a1h-thr 0.5 --a1t-thr 0.5 \
  --outdir ./results/attack2/ \
  --device cuda --seed 42

# Attack 3 - Graph reconstruction
python attacks/attack3.py \
  --public_tsv ./prepared_data/public.tsv \
  --sensitive_dir ./prepared_data/sensitive/ \
  --sensitive_files rel1.tsv,rel2.tsv \
  --feature-combination "all" \
  --knn_k 120 --max-layer 2 \
  --outdir ./results/attack3/ --seed 42
```

### Step 3: Apply Defenses

```bash
# K-Anonymity defense
bash defenses/kanon_runner.sh

# Randomized Response defense
bash defenses/rr_runner.sh

# CHAMELEON adaptive defense
bash defenses/chameleon_runner.sh
```

### Step 4: Evaluate Utility

```bash
python experiments/utility_LinkPrediction.py \
  --public-path ./prepared_data/public.tsv \
  --defended-path ./results/defended_graph.tsv
```
### Step 5: Run the features study

```bash
bash experiments/runner1.sh    # All Attack 1 experiments
bash experiments/runner2.sh    # All Attack 2 experiments
bash experiments/runner3.sh    # All Attack 3 experiments
```

## 🛡️ Defense Details

### K-Anonymity Defense
Modifies the graph to ensure each node has k-anonymity among sensitive entities.
```bash
bash defenses/kanon_runner.sh
```
**Parameters:** k-value (default: 5), modification budget

### Randomized Response Defense
Adds probabilistic noise to sensitive triples.
```bash
bash defenses/rr_runner.sh
```
**Parameters:** Noise probability (epsilon), privacy budget

### CHAMELEON Defense
Adaptively selects defense strategy per node based on structural profile.
```bash
bash defenses/chameleon_runner.sh
```
**Parameters:** Budget allocation, node profile thresholds

## 📈 Evaluation Metrics

**Privacy Metrics:**
| Metric | Description |
|--------|-------------|
|PR-AUC | Precision-recall trade-off for attacks |
| AUC-ROC | Area under ROC curve for binary classification |
| Hit@K | Fraction of ground truth in top-K predictions |

**Utility Metrics:**
| Metric | Description |
|--------|-------------|
| Link Prediction AUC | Utility of defended graph for downstream tasks |
| Degree Distribution | Preservation of degree statistics |
| Clustering Coefficient | Graph local connectivity preservation |

## 🔄 Recommended Workflow

1. **Baseline Assessment**
   ```bash
   # Prepare data for analysis
   python dataprocessing/split.py --global_path <kg> --relation <rel> --outdir ./data
   ```

2. **Run Attacks (Undefended)**
   ```bash
   # Measure vulnerability on original public graph
   bash experiments/runner1.sh
   bash experiments/runner2.sh
   bash experiments/runner3.sh
   ```

3. **Apply Defenses**
   ```bash
   # Protect the graph
   bash defenses/kanon_runner.sh
   bash defenses/rr_runner.sh
   bash defenses/chameleon_runner.sh
   ```

4. **Evaluate Privacy-Utility Trade-off**
   ```bash
   # Measure utility on defended graphs
   python experiments/utility_LinkPrediction.py \
     --public-path ./data/public.tsv \
     --defended-path ./results/defended.tsv
   ```

5. **Feature Analysis (Optional)**
   ```bash
   # Understand which features drive attacks
   python experiments/attack1_featuresstudy.py --public-path <path> --sens-path <path>
   python experiments/attack2_featuresstudy.py --public-path <path> --sens-path <path>
   python experiments/attack3_featuresstudy.py --public-path <path> --sens-dir <path>
   ```

## ⚡ Minimal Working Example

Quickly reproduce Figure 3 (NELL dataset):

```bash
# 1. Prepare sample data
python dataprocessing/split.py \
  --global_path nell_sample.tsv \
  --relation concept:teamplaysagainstteam \
  --outdir ./nell_out/

# 2. Run Attack 1 (5-10 min)
python attacks/attack1_head.py \
  --public-path ./nell_out/public.tsv \
  --sens-path ./nell_out/concept:teamplaysagainstteam.tsv \
  --outdir ./results/attack1/ \
  --seed 42

# 3. Apply defense (2-5 min)
bash defenses/kanon_runner.sh

# 4. Measure utility
python experiments/utility_LinkPrediction.py \
  --public-path ./nell_out/public.tsv \
  --defended-path ./results/defended.tsv
```


## 🔧 Tech Stack

- **PyTorch 2.4.1** – Neural network models (MLP classifiers)
- **NetworkX 3.1** – Graph algorithms & topology analysis
- **Scikit-learn 1.3.2** – Machine learning utilities, metrics
- **Pandas 2.0.3 / NumPy 1.24.4** – Data manipulation & numerical computing

```

## 📞 Support & Issues

For questions, bug reports, or feature requests, please open an issue on GitHub or contact the maintainers.

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).


