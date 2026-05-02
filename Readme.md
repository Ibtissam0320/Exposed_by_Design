# Exposed by Design: Privacy Attacks and Defenses for Knowledge Graphs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4.1-red)](https://pytorch.org/)

This repository contains the code and resources to reproduce the main results of the paper **"Exposed by Design: Privacy Attacks and Defenses for Knowledge Graphs"**.

---

## Requirements and Setup

To set up the environment and install all required libraries:

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. The main requirements include:
   - PyTorch 2.4.1
   - NetworkX 3.1
   - Scikit-learn 1.3.2
   - Pandas 2.0.3
   - NumPy 1.24.4
   - CUDA (optional, for GPU acceleration)

---

## Hardware Requirements

Our results were primarily produced using the following setup:

- NVIDIA GPU (RTX 3090 or A100, 24 GB VRAM)
- 16+ GB RAM
- 8-core CPU
- Ubuntu 20.04 or later

We have also tested the code on a smaller CPU-only setup:

- 8 GB RAM
- 4-core CPU
- Any OS with Python 3.7+

**Note:** Experiments on NELL can be run on CPU in approximately 1–2 hours. GPU acceleration is recommended for FB15k-237 and HealthKG.

---

## Data Preparation

Download the datasets from the following sources:

- **NELL**: https://rtw.ml.cmu.edu/rtw/
- **FB15k-237**: https://huggingface.co/datasets/KGraph/FB15k-237
- **HealthKG**: https://github.com/Boreico/KGE_QCB_Project

Once downloaded, split each dataset into public and sensitive subgraphs using the provided utility:

```bash
python dataprocessing/split.py \
  --global_path /path/to/full_kg.tsv \
  --relation "sensitive_relation_1" \
  --relation "sensitive_relation_2" \
  --outdir /path/to/output/
```

Refer to `dataprocessing/README.md` for the full list of sensitive relations used per dataset and expected directory structure.

---

## Reproducing Results

To reproduce the results of the paper, follow these steps:

1. Prepare the data as described above.
2. Navigate to the `experiments/` directory and run the appropriate runner script for each attack.
3. To apply defenses, run the corresponding scripts in the `defenses/` directory.
4. After execution, results are saved in the `results/` directory.

### Recommended Workflow

1. Prepare the datasets using `dataprocessing/split.py`.
2. Run the undefended attack experiments:

```bash
python attacks/attack1_head.py \
  --public-path ./nell_out/public.tsv \
  --sens-path ./nell_out/concept:teamplaysagainstteam.tsv \
  --outdir ./results/attack1/ \
  --seed 42    # Attack 1
python attack2_pairwise.py \
  --public-path data/public.tsv \
  --sens-path data/sensitive.tsv \
  --attack1-head-scores results/attack1_heads_scores.tsv \
  --attack1-tail-scores results/attack1_tails_scores.tsv \
  --outdir results_attack2/   # Attack 2
python attack3_knn_reconstruction.py \
  --public_tsv data/public.tsv \
  --sensitive_dir data/sensitive \
  --sensitive_files has_age_category.tsv \
  --outdir results_attack3 \
  --seed_frac 0.2 \
  --knn_k 120 \
  --max_pred_per_head 3 \
  --vote_threshold 0.0 \
  --hash_dim_out 256 \    # Attack 3
```

3. Apply the defense mechanisms:

```bash
bash defenses/kanon_runner.sh       # K-Anonymity
bash defenses/rr_runner.sh          # Randomized Response
bash defenses/chameleon_runner.sh   # CHAMELEON
```

4. Evaluate the privacy-utility trade-off:

```bash
python experiments/utility_LinkPrediction.py \
  --public-path ./prepared_data/public.tsv \
  --defended-path ./results/defended_graph.tsv
```

5. Run the feature ablation studies:

```bash
bash experiments/runner1.sh   # Attack 1 ablation
bash experiments/runner2.sh   # Attack 2 ablation
bash experiments/runner3.sh   # Attack 3 ablation
```

---

## Experiments

The main results of the paper are organized as follows:

- **Attack 1 — Link Inference:** Predicts whether a head vertex is involoved in a target relation using topological features from the public graph.
- **Attack 2 — Triple Inference:** Given a confirmed target head entity, infers the most likely tail entity for the target relation.
- **Attack 3 — Link Prediction via Structural Propagation:** Predicts sensitive triples by propagating labels from a seed set using k-nearest neighbor similarity over structural feature vectors.

Each attack is associated with:

1. An implementation script in the `attacks/` directory.
2. A batch runner in the `experiments/` directory.
3. A feature ablation script in the `experiments/` directory.

---

## Minimal Working Example: Reproduce Attack 1 on NELL

Follow these steps to quickly reproduce the Attack 1 results on the NELL dataset.

1. Prepare the data:

```bash
python dataprocessing/split.py \
  --global_path nell_sample.tsv \
  --relation concept:teamplaysagainstteam \
  --outdir ./nell_out/
```

2. Run Attack 1 (approximately 5–10 minutes):

```bash
python attacks/attack1_head.py \
  --public-path ./nell_out/public.tsv \
  --sens-path ./nell_out/concept:teamplaysagainstteam.tsv \
  --outdir ./results/attack1/ \
  --seed 42
```

3. Apply the K-Anonymity defense:

```bash
bash defenses/kanon_runner.sh
```

4. Evaluate utility:

```bash
python experiments/utility_LinkPrediction.py \
  --public-path ./nell_out/public.tsv \
  --defended-path ./results/defended.tsv
```

---

## Directory Structure

```
.
├── attacks/                            # Attack implementations
│   ├── attack1_head.py
│   ├── attack1_tail.py
│   ├── attack2.py
│   └── attack3.py
│
├── defenses/                           # Defense mechanisms
│   ├── defense_kanonymity.py
│   ├── defense_randomized_response.py
│   ├── chameleon_defense.py
│   ├── kanon_runner.sh
│   ├── rr_runner.sh
│   └── chameleon_runner.sh
│
├── dataprocessing/
│   ├── split.py
│   └── README.md
│
├── experiments/                        # Ablation studies and evaluation
│   ├── attack1_featuresstudy.py
│   ├── attack2_featuresstudy.py
│   ├── attack3_featuresstudy.py
│   ├── utility_LinkPrediction.py
│   ├── runner1.sh
│   ├── runner2.sh
│   └── runner3.sh
│
├── requirements.txt
└── README.md
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

For questions or issues related to this repository, please open a GitHub issue or contact the authors directly.