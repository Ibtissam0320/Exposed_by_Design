#!/usr/bin/env python3
"""
Attack 1 - kNN head scorer
2 features: out_deg + Ri_L1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def load_public_graph(public_path: Path, has_header: bool):
    print(f"[+] Loading public triples from {public_path}")
    if has_header:
        df = pd.read_csv(public_path, sep="\t", header=0,
                         names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    else:
        df = pd.read_csv(public_path, sep="\t", header=None,
                         names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    print(f"    Loaded {len(df):,} public triples")

    out_neighbors   = defaultdict(set)
    neighbors_w_rel = defaultdict(list)

    for row in df.itertuples(index=False):
        h, r, t = str(row.head_id), str(row.rel_id), str(row.tail_id)
        out_neighbors[h].add(t)
        neighbors_w_rel[h].append((t, r))
        neighbors_w_rel[t].append((h, r))

    print(f"[+] Public graph: {len(neighbors_w_rel):,} nodes")
    return out_neighbors, neighbors_w_rel


def load_sensitive_heads(sens_path: Path):
    print(f"[+] Loading sensitive heads from {sens_path}")
    df = pd.read_csv(sens_path, sep="\t", header=None,
                     names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    pos = set(df["head_id"].astype(str).unique())
    print(f"    Positive heads: {len(pos):,}")
    return pos


def ri_l1(node, neighbors_w_rel):
    rels = set(r for _, r in neighbors_w_rel.get(str(node), []))
    return float(len(rels))


def build_features(node_ids, out_neighbors, neighbors_w_rel):
    print(f"[+] Building features for {len(node_ids):,} nodes")
    X = np.array([
        [float(len(out_neighbors.get(str(n), set()))),
         ri_l1(n, neighbors_w_rel)]
        for n in node_ids
    ], dtype=float)
    print(f"    Feature matrix: {X.shape}  [out_deg, Ri_L1]")
    return X


def knn_classify(X_train, y_train, X_test, k=50):
    """
    kNN with cosine similarity.
    Train set must have balanced pos/neg ratio for kNN to work.
    """
    print(f"\n[+] kNN (k={k}, cosine)")
    scaler    = StandardScaler()
    X_tr      = scaler.fit_transform(X_train)
    X_te      = scaler.transform(X_test)
    X_tr_norm = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_te_norm = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)

    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X_tr_norm)
    distances, indices = nn.kneighbors(X_te_norm)
    sims = 1.0 - distances

    scores = []
    for i in range(len(X_test)):
        labels = y_train[indices[i]]
        w      = sims[i]
        pos_w  = np.sum(w[labels == 1])
        total  = pos_w + np.sum(w[labels == 0])
        scores.append(pos_w / total if total > 0 else 0.5)
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-path",        required=True)
    ap.add_argument("--sens-path",          required=True)
    ap.add_argument("--outdir",             required=True)
    ap.add_argument("--k",                  type=int,   default=50)
    # neg-ratio controls train set balance: neg = pos * neg-ratio
    ap.add_argument("--neg-ratio",          type=int,   default=5,
                    help="Train negatives = n_train_pos * neg_ratio (keeps balance)")
    ap.add_argument("--test-neg-sample",    type=int,   default=200)
    ap.add_argument("--pos-train-fraction", type=float, default=0.2)
    ap.add_argument("--public-has-header",  action="store_true")
    ap.add_argument("--seed",               type=int,   default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out_neighbors, neighbors_w_rel = load_public_graph(
        Path(args.public_path), args.public_has_header)

    pos_heads = load_sensitive_heads(Path(args.sens_path))
    all_nodes = sorted(neighbors_w_rel.keys())
    pos_list  = sorted(pos_heads)
    neg_list  = [n for n in all_nodes if n not in pos_heads]

    # Split positives
    n_train_pos = max(1, int(args.pos_train_fraction * len(pos_list)))
    perm        = rng.permutation(len(pos_list))
    train_pos   = [pos_list[i] for i in perm[:n_train_pos]]
    test_pos    = [pos_list[i] for i in perm[n_train_pos:]]

    # BALANCED train negatives: n_train_pos * neg_ratio
    n_train_neg = min(n_train_pos * args.neg_ratio, len(neg_list))
    train_neg   = rng.choice(neg_list, size=n_train_neg, replace=False)

    # Test negatives can be large
    test_neg = rng.choice(neg_list,
                          size=min(args.test_neg_sample, len(neg_list)),
                          replace=False)

    # Adaptive k: never exceed train set size
    k = min(args.k, len(train_pos) + len(train_neg) - 1)

    nodes_train = list(train_pos) + list(train_neg)
    y_train     = np.array([1]*len(train_pos) + [0]*len(train_neg))
    nodes_test  = list(test_pos)  + list(test_neg)
    y_test      = np.array([1]*len(test_pos)  + [0]*len(test_neg))

    print(f"[+] Train: {len(train_pos)} pos, {len(train_neg)} neg  (ratio 1:{args.neg_ratio})")
    print(f"[+] Test:  {len(test_pos)} pos, {len(test_neg)} neg")
    print(f"[+] k = {k}")

    X_train = build_features(nodes_train, out_neighbors, neighbors_w_rel)
    X_test  = build_features(nodes_test,  out_neighbors, neighbors_w_rel)

    y_scores = knn_classify(X_train, y_train, X_test, k=k)

    pr_auc  = float(average_precision_score(y_test, y_scores))
    try:
        roc_auc = float(roc_auc_score(y_test, y_scores))
    except:
        roc_auc = None

    print(f"\n✅ PR-AUC  = {pr_auc:.4f}")
    if roc_auc:
        print(f"✅ ROC-AUC = {roc_auc:.4f}")

    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sens_name = Path(args.sens_path).stem

    pd.DataFrame({"head_id": nodes_test, "label": y_test, "score": y_scores}) \
      .to_csv(outdir / f"{sens_name}_knn_ri_scores.tsv", sep="\t", index=False)

    with open(outdir / f"{sens_name}_knn_ri_metrics.json", "w") as f:
        json.dump({
            "method":       "kNN_cosine_out_deg_Ri_L1",
            "k":            k,
            "neg_ratio":    args.neg_ratio,
            "num_features": 2,
            "features":     ["out_deg", "Ri_L1"],
            "pr_auc":       pr_auc,
            "roc_auc":      roc_auc,
            "train_pos":    len(train_pos),
            "train_neg":    len(train_neg),
            "test_pos":     len(test_pos),
            "test_neg":     len(test_neg),
        }, f, indent=2)

    print(f"✅ Saved → {outdir}")


if __name__ == "__main__":
    main()