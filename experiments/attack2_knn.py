#!/usr/bin/env python3
"""
Attack 2 - kNN ONLY
Features: ni_head_L1, ni_tail_L1, Ii_head_L1, Ii_tail_L1, Ei_head_L1, Ei_tail_L1
Metric: MRR, Hits@1, Hits@3, Hits@5, Hits@10
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def load_public_graph(path: Path, has_header: bool):
    print(f"[+] Loading public graph: {path}")
    df = pd.read_csv(path, sep="\t", header=0 if has_header else None,
                     names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    print(f"    Triples: {len(df):,}")
    neighbors = defaultdict(set)
    for row in df.itertuples(index=False):
        h, t = str(row.head_id), str(row.tail_id)
        neighbors[h].add(t)
        neighbors[t].add(h)
    print(f"    Nodes: {len(neighbors):,}")
    return neighbors


def load_sensitive_triples(path: Path):
    print(f"[+] Loading sensitive triples: {path}")
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    print(f"    Triples: {len(df):,}")
    return df


def load_attack1_scores(path: Path, id_col: str, score_col: str):
    df = pd.read_csv(path, sep="\t", dtype=str).dropna()
    df[id_col]    = df[id_col].astype(str)
    df[score_col] = df[score_col].astype(float)
    return dict(zip(df[id_col], df[score_col]))


def bfs_layer1(node, neighbors):
    return neighbors.get(str(node), set())

def ni_L1(node, neighbors):
    return float(len(bfs_layer1(node, neighbors)))

def Ii_L1(node, neighbors):
    layer = list(bfs_layer1(node, neighbors))
    if len(layer) < 2:
        return 0.0
    if len(layer) > 100:
        layer = layer[:100]
    edges = sum(1 for i, u in enumerate(layer)
                for v in layer[i+1:]
                if v in neighbors.get(str(u), set()))
    max_e = len(layer) * (len(layer) - 1) / 2
    return float(edges / max_e) if max_e > 0 else 0.0

def Ei_L1(node, neighbors):
    layer1 = bfs_layer1(node, neighbors)
    seen   = {str(node)} | layer1
    layer2 = set()
    for u in layer1:
        for v in neighbors.get(str(u), set()):
            if v not in seen:
                layer2.add(v)
    return float(len(layer2))

def extract_features(h, t, neighbors):
    return np.array([
        ni_L1(h, neighbors), ni_L1(t, neighbors),
        Ii_L1(h, neighbors), Ii_L1(t, neighbors),
        Ei_L1(h, neighbors), Ei_L1(t, neighbors),
    ], dtype=np.float32)


def knn_score(X_train, y_train, X_test, k=50):
    print(f"[+] kNN (k={k}, cosine)")
    scaler    = StandardScaler()
    X_tr      = scaler.fit_transform(X_train)
    X_te      = scaler.transform(X_test)
    X_tr_norm = X_tr / (np.linalg.norm(X_tr, axis=1, keepdims=True) + 1e-8)
    X_te_norm = X_te / (np.linalg.norm(X_te, axis=1, keepdims=True) + 1e-8)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X_tr_norm)
    dists, idxs = nn.kneighbors(X_te_norm)
    sims = 1.0 - dists
    scores = []
    for i in range(len(X_test)):
        w     = sims[i]
        lbs   = y_train[idxs[i]]
        pos_w = np.sum(w[lbs == 1])
        total = np.sum(w)
        scores.append(pos_w / total if total > 0 else 0.5)
    return np.array(scores, dtype=np.float32)


def compute_mrr(heads_test, tails_test, y_test, y_scores, ks=[1,3,5,10]):
    """Compute MRR and Hits@K per head."""
    # Group by head
    head2pairs = defaultdict(list)
    for h, t, label, score in zip(heads_test, tails_test, y_test, y_scores):
        head2pairs[h].append((t, int(label), float(score)))

    mrr_list = []
    hits = {k: [] for k in ks}

    for h, pairs in head2pairs.items():
        # Sort by score descending
        pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
        # Find rank of first positive
        rank = None
        for i, (t, label, score) in enumerate(pairs_sorted, start=1):
            if label == 1:
                rank = i
                break
        if rank is None:
            mrr_list.append(0.0)
            for k in ks:
                hits[k].append(0.0)
        else:
            mrr_list.append(1.0 / rank)
            for k in ks:
                hits[k].append(1.0 if rank <= k else 0.0)

    mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    hits_out = {f"hits@{k}": float(np.mean(hits[k])) for k in ks}
    return mrr, hits_out, len(mrr_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-path",          required=True)
    ap.add_argument("--sens-path",            required=True)
    ap.add_argument("--attack1-head-scores",  required=True)
    ap.add_argument("--attack1-tail-scores",  required=True)
    ap.add_argument("--outdir",               required=True)
    ap.add_argument("--a1h-id-col",    default="head_id")
    ap.add_argument("--a1h-score-col", default="score")
    ap.add_argument("--a1t-id-col",    default="tail_id")
    ap.add_argument("--a1t-score-col", default="score")
    ap.add_argument("--a1h-thr",       type=float, default=0.0)
    ap.add_argument("--a1t-thr",       type=float, default=0.0)
    ap.add_argument("--neg-per-pos",   type=int,   default=10)
    ap.add_argument("--k",             type=int,   default=50)
    ap.add_argument("--public-has-header", action="store_true")
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    neighbors  = load_public_graph(Path(args.public_path), args.public_has_header)
    df_sens    = load_sensitive_triples(Path(args.sens_path))
    a1h_scores = load_attack1_scores(Path(args.attack1_head_scores),
                                     args.a1h_id_col, args.a1h_score_col)
    a1t_scores = load_attack1_scores(Path(args.attack1_tail_scores),
                                     args.a1t_id_col, args.a1t_score_col)

    cand_heads = set(h for h, s in a1h_scores.items() if s >= args.a1h_thr)
    cand_tails = set(t for t, s in a1t_scores.items() if s >= args.a1t_thr)
    tail_pool  = np.array(sorted(cand_tails), dtype=object)

    print(f"[+] Candidate heads: {len(cand_heads):,}  tails: {len(cand_tails):,}")

    # Build pairs — one positive per head (aligned with MLP ablation)
    # Group all true tails per head first
    head2tails = defaultdict(list)
    for row in df_sens.itertuples(index=False):
        h, t = str(row.head_id), str(row.tail_id)
        if h in cand_heads and t in cand_tails:
            head2tails[h].append(t)

    X_all, y_all, heads_all, tails_all = [], [], [], []
    for h, true_tails in head2tails.items():
        # Pick one positive tail per head (first one)
        t = true_tails[0]
        X_all.append(extract_features(h, t, neighbors))
        y_all.append(1)
        heads_all.append(h); tails_all.append(t)
        for _ in range(args.neg_per_pos):
            neg_t = str(rng.choice(tail_pool))
            if neg_t in true_tails:
                continue
            X_all.append(extract_features(h, neg_t, neighbors))
            y_all.append(0)
            heads_all.append(h); tails_all.append(neg_t)

    X_all     = np.array(X_all,     dtype=np.float32)
    y_all     = np.array(y_all,     dtype=int)
    heads_all = np.array(heads_all, dtype=object)
    tails_all = np.array(tails_all, dtype=object)

    print(f"[+] Total pairs: {len(y_all):,}  (pos={y_all.sum():,})")
    print(f"[+] Features: {X_all.shape[1]}  [ni_h, ni_t, Ii_h, Ii_t, Ei_h, Ei_t]")

    # Train/test split
    idx = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3,
                                      random_state=args.seed, stratify=y_all)
    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_test,  y_test  = X_all[te_idx], y_all[te_idx]
    heads_test = heads_all[te_idx]
    tails_test = tails_all[te_idx]

    print(f"[+] Train: {len(y_train):,}  Test: {len(y_test):,}")

    y_scores = knn_score(X_train, y_train, X_test, k=args.k)

    mrr, hits_out, n_heads = compute_mrr(heads_test, tails_test, y_test, y_scores)

    print(f"\n✅ MRR      = {mrr:.4f}")
    for k, v in hits_out.items():
        print(f"✅ {k:8s} = {v:.4f}")
    print(f"   Heads evaluated: {n_heads}")

    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sens_name = Path(args.sens_path).stem

    pd.DataFrame({
        "head_id": heads_test, "tail_id": tails_test,
        "label": y_test, "score": y_scores
    }).to_csv(outdir / f"{sens_name}_attack2_knn_scores.tsv", sep="\t", index=False)

    with open(outdir / f"{sens_name}_attack2_knn_metrics.json", "w") as f:
        json.dump({
            "method":       "attack2_kNN_cosine_ni_Ii_Ei_L1",
            "k":            args.k,
            "num_features": int(X_all.shape[1]),
            "features":     ["ni_h_L1","ni_t_L1","Ii_h_L1","Ii_t_L1","Ei_h_L1","Ei_t_L1"],
            "mrr":          mrr,
            **hits_out,
            "n_heads":      n_heads,
            "train_pairs":  int(len(y_train)),
            "test_pairs":   int(len(y_test)),
        }, f, indent=2)

    print(f"✅ Saved → {outdir}")


if __name__ == "__main__":
    main()