#!/usr/bin/env python3
"""
Attack 1 (TAIL-WISE, LOCAL ONLY) for Health_KG.

Binary MLP for EXISTENCE of a sensitive relation per TAIL node.

LOCAL features only:
  - undirected degree
  - relation diversity
  - average neighbor degree
  - per-relation undirected counts

NO:
  - hard negatives
  - hashed features
  - cycle proxy
  - kNN
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Load public graph (UNDIRECTED ONLY)
# ---------------------------------------------------------
def load_public_graph(public_path: Path, has_header: bool):
    print(f"[+] Loading public triples from {public_path}")

    df = pd.read_csv(
        public_path,
        sep="\t",
        header=0 if has_header else None,
        names=["head_id", "rel_id", "tail_id"],
        dtype=str,
        low_memory=False,
    ).dropna()

    neighbors = defaultdict(set)
    rel_deg = defaultdict(lambda: defaultdict(int))

    for h, r, t in df.itertuples(index=False):
        neighbors[h].add(t)
        neighbors[t].add(h)
        rel_deg[h][r] += 1
        rel_deg[t][r] += 1

    rel_ids = sorted(df["rel_id"].unique().tolist())
    print(f"[+] Public graph: nodes={len(neighbors):,}, relations={len(rel_ids)}")
    return neighbors, rel_deg, rel_ids


# ---------------------------------------------------------
# Load sensitive TAILS (ground truth)
# ---------------------------------------------------------
def load_sensitive_tails(sens_path: Path):
    print(f"[+] Loading sensitive triples from {sens_path}")
    df = pd.read_csv(
        sens_path,
        sep="\t",
        header=None,
        names=["head_id", "rel_id", "tail_id"],
        dtype=str,
        low_memory=False,
    ).dropna()

    pos_tails = set(df["tail_id"].unique().tolist())
    print(f"    #positive tails = {len(pos_tails):,}")
    return pos_tails


# ---------------------------------------------------------
# Build LOCAL structural features for tails
# ---------------------------------------------------------
def build_tail_features(tail_ids, neighbors, rel_deg, rel_ids, degrees):
    print(f"[+] Building features for {len(tail_ids):,} tails")
    X = []

    for t in tail_ids:
        t = str(t)
        deg = degrees.get(t, 0)
        rel_dict = rel_deg[t]
        rel_div = len(rel_dict)

        neighs = neighbors.get(t, set())
        if deg > 0:
            avg_nb_deg = np.mean([degrees.get(n, 0) for n in neighs])
        else:
            avg_nb_deg = 0.0

        rel_feats = [rel_dict.get(r, 0) for r in rel_ids]

        X.append([deg, rel_div, avg_nb_deg] + rel_feats)

    X = np.asarray(X, dtype=float)
    print(f"    Feature matrix shape = {X.shape}")
    return X


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--public-path", type=str, default= "/home/harrouch/Nell/processed/global_kg_public_wo_sensitive.tsv")
    ap.add_argument("--sens-path", type=str,default="/home/harrouch/Nell/processed/sensitive/concept:proxyfor.tsv")
    ap.add_argument("--public-has-header", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-train-fraction", type=float, default=0.2)
    ap.add_argument("--train-neg-sample", type=int, default=5000)
    ap.add_argument("--test-neg-sample", type=int, default=5000)

    ap.add_argument("--outdir", type=str, default="/home/harrouch/Nell")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    scores_dir = outdir / "scores_20pct"
    metrics_dir = outdir / "metrics_20pct"
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    neighbors, rel_deg, rel_ids = load_public_graph(
        Path(args.public_path), args.public_has_header
    )
    degrees = {n: len(v) for n, v in neighbors.items()}

    pos_tails_all = np.array(sorted(load_sensitive_tails(Path(args.sens_path))), dtype=object)
    all_nodes = np.array(sorted(neighbors.keys()), dtype=object)

    neg_tails_all = np.array([n for n in all_nodes if n not in set(pos_tails_all)], dtype=object)

    print(f"[+] Total tails: pos={len(pos_tails_all):,}, neg={len(neg_tails_all):,}")

    # Split positives
    perm = rng.permutation(len(pos_tails_all))
    k = max(1, int(args.pos_train_fraction * len(pos_tails_all)))

    train_pos = pos_tails_all[perm[:k]]
    test_pos = pos_tails_all[perm[k:]]

    # Negatives
    train_unlabeled = np.array([n for n in all_nodes if n not in set(train_pos)], dtype=object)
    train_neg = rng.choice(train_unlabeled, size=min(args.train_neg_sample, len(train_unlabeled)), replace=False)
    test_neg = rng.choice(neg_tails_all, size=min(args.test_neg_sample, len(neg_tails_all)), replace=False)

    # Build datasets
    X_train = build_tail_features(
        np.concatenate([train_pos, train_neg]), neighbors, rel_deg, rel_ids, degrees
    )
    y_train = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])

    X_test = build_tail_features(
        np.concatenate([test_pos, test_neg]), neighbors, rel_deg, rel_ids, degrees
    )
    y_test = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])

    # Train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, random_state=args.seed)
    clf.fit(X_train, y_train)

    # Eval
    scores = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)

    print(f"[+] PR-AUC = {pr_auc:.4f}")
    print(f"[+] ROC-AUC = {roc_auc:.4f}")

    # Save
    df_scores = pd.DataFrame({
        "tail_id": np.concatenate([test_pos, test_neg]),
        "label": y_test,
        "score": scores
    })
    df_scores.to_csv(scores_dir / "tail_existence_scores.tsv", sep="\t", index=False)

    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump({
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "num_train_pos": int(len(train_pos)),
            "num_test_pos": int(len(test_pos)),
            "num_features": int(X_train.shape[1]),
        }, f, indent=2)

    print("✅ Attack1 Health_KG TAIL-WISE LOCAL DONE")


if __name__ == "__main__":
    main()
