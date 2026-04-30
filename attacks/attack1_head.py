#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack 1 (Head-level) — Sensitive relation existence inference using ONLY public graph structure.

Inputs:
  - Public graph TSV:   head_id \t rel_id \t tail_id   (Wikidata-style Q*/P*)
  - Sensitive TSV (e.g., P140): head_id \t rel_id \t tail_id

Logic (same as Health_KG/Synthea versions):
  1) Load public graph and compute structural stats (undirected + directed)
  2) Positives = heads appearing in sensitive TSV; negatives = public nodes minus positives
  3) Split positives: train_pos (seeds) vs test_pos (hidden positives)
  4) Train negatives sampled from UNLABELED pool (non-seeds)  (may contain hidden positives)
  5) Test negatives sampled from TRUE negatives pool
  6) Build structural features + optional cycle proxy + optional kNN
  7) Train sklearn MLP, evaluate PR-AUC/ROC-AUC
  8) Save scores + metrics

Notes:
  - Uses string IDs (Q..., P...)
  - "Closed world" labels: heads not in sensitive file => label 0 (true negatives pool)
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

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


# ---------------------------------------------------------
# Load public graph (directed + undirected stats)
# ---------------------------------------------------------
def load_public_graph(public_path: Path, has_header: bool):
    print(f"[+] Loading public triples from {public_path}")

    if has_header:
        df_pub = pd.read_csv(
            public_path,
            sep="\t",
            header=0,
            names=["head_id", "rel_id", "tail_id"],
            dtype=str,
            low_memory=False,
        ).dropna()
    else:
        df_pub = pd.read_csv(
            public_path,
            sep="\t",
            header=None,
            names=["head_id", "rel_id", "tail_id"],
            dtype=str,
            low_memory=False,
        ).dropna()

    print(f"    Loaded {len(df_pub):,} public triples")

    neighbors_und = defaultdict(set)
    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)

    rel_deg_total = defaultdict(lambda: defaultdict(int))
    rel_deg_out = defaultdict(lambda: defaultdict(int))
    rel_deg_in = defaultdict(lambda: defaultdict(int))

    for row in df_pub.itertuples(index=False):
        h = str(row.head_id)
        r = str(row.rel_id)
        t = str(row.tail_id)

        out_neighbors[h].add(t)
        in_neighbors[t].add(h)
        rel_deg_out[h][r] += 1
        rel_deg_in[t][r] += 1

        neighbors_und[h].add(t)
        neighbors_und[t].add(h)
        rel_deg_total[h][r] += 1
        rel_deg_total[t][r] += 1

    rel_ids = sorted(df_pub["rel_id"].astype(str).unique().tolist())
    print(f"[+] Public graph: {len(neighbors_und):,} nodes, {len(rel_ids)} distinct relations")
    return neighbors_und, out_neighbors, in_neighbors, rel_deg_total, rel_deg_out, rel_deg_in, rel_ids


def load_sensitive_heads(sens_path: Path):
    print(f"[+] Loading sensitive triples from {sens_path}")
    df_sens = pd.read_csv(
        sens_path,
        sep="\t",
        header=None,
        names=["head_id", "rel_id", "tail_id"],
        dtype=str,
        low_memory=False,
    ).dropna()
    print(f"    Loaded {len(df_sens):,} sensitive triples")
    pos_heads_all = set(df_sens["head_id"].astype(str).unique().tolist())
    rel = df_sens["rel_id"].iloc[0] if len(df_sens) > 0 else ""
    print(f"    #unique sensitive heads (positives): {len(pos_heads_all):,}")
    return pos_heads_all, str(rel)


# ---------------------------------------------------------
# Hard negative selection modes
# ---------------------------------------------------------
def hardneg_select(mode, neg_candidates, degrees_total, ref_pos, num_sample, rng, band_alpha):
    neg_candidates = list(neg_candidates)

    if mode == "bruteforce_all":
        return np.array(neg_candidates, dtype=object), None, len(neg_candidates)

    pos_degs = np.array([degrees_total.get(str(h), 0.0) for h in ref_pos], dtype=float)
    if len(pos_degs) == 0:
        raise RuntimeError("No positives provided for hard negative selection")

    m = float(np.median(pos_degs))

    if mode == "median_ge":
        pool = [h for h in neg_candidates if degrees_total.get(str(h), 0.0) >= m]
    elif mode == "median_band":
        lo, hi = m - float(band_alpha), m + float(band_alpha)
        pool = [h for h in neg_candidates if lo <= degrees_total.get(str(h), 0.0) <= hi]
    elif mode == "none":
        pool = neg_candidates
    else:
        raise ValueError(f"Unknown hardneg mode: {mode}")

    if len(pool) == 0:
        pool = neg_candidates

    k = min(int(num_sample), len(pool))
    sampled = rng.choice(np.array(pool, dtype=object), size=k, replace=False)
    return sampled, m, len(pool)


# ---------------------------------------------------------
# Cycle proxy (light) among 1-hop neighbors (optional)
# ---------------------------------------------------------
def cycle_proxy_light_1hop(h, out_neighbors, in_neighbors, neighbors_und, rng, sample_size):
    if sample_size is None or int(sample_size) <= 0:
        return 0.0

    S1 = set()
    S1 |= out_neighbors.get(h, set())
    S1 |= in_neighbors.get(h, set())
    if len(S1) == 0:
        return 0.0

    S1_list = np.array(list(S1), dtype=object)
    k = min(int(sample_size), len(S1_list))
    sampled = rng.choice(S1_list, size=k, replace=False)
    Sset = set(map(str, sampled.tolist()))

    edge_cnt2 = 0
    for u in sampled:
        Au = neighbors_und.get(str(u), set())
        edge_cnt2 += sum((1 for v in Sset if v in Au))

    edges = edge_cnt2 / 2.0
    return float(edges / max(1.0, float(len(Sset))))


# ---------------------------------------------------------
# Hashed predicate profile (direction-aware)
# ---------------------------------------------------------
def hashed_predicate_profile(rel_dict, D, deg_norm):
    if D <= 0:
        return []
    v = np.zeros(int(D), dtype=float)
    if rel_dict:
        for r, c in rel_dict.items():
            j = (hash(str(r)) % int(D))
            v[j] += float(c)
    denom = max(1.0, float(deg_norm))
    v /= denom
    return v.tolist()


# ---------------------------------------------------------
# Structural features (extended, same style as others)
# ---------------------------------------------------------
def build_head_features(
    head_ids,
    neighbors_und,
    out_neighbors,
    in_neighbors,
    rel_deg_total,
    rel_deg_out,
    rel_deg_in,
    rel_ids,
    deg_total,
    deg_out,
    deg_in,
    hashed_d_out=64,
    hashed_d_in=64,
    add_cycle_proxy=False,
    cycle_sample_size=50,
    rng_seed=42,
):
    head_ids = list(head_ids)
    print(f"[+] Building head-level features for {len(head_ids):,} heads")
    rng = np.random.default_rng(int(rng_seed))

    X_rows = []
    for h in head_ids:
        h = str(h)

        d_tot = float(deg_total.get(h, 0))
        rel_dict_tot = rel_deg_total[h]
        rel_div_tot = float(len(rel_dict_tot))

        neighs_und = neighbors_und.get(h, set())
        if d_tot > 0 and len(neighs_und) > 0:
            nb_degs = [deg_total.get(str(v), 0) for v in neighs_und]
            avg_nb_deg = float(np.mean(nb_degs)) if len(nb_degs) > 0 else 0.0
        else:
            avg_nb_deg = 0.0

        rel_feats_total = [float(rel_dict_tot.get(r, 0)) for r in rel_ids]

        d_out = float(deg_out.get(h, 0))
        d_in = float(deg_in.get(h, 0))
        d = d_out + d_in

        delta_out = float(len(rel_deg_out[h]))
        delta_in = float(len(rel_deg_in[h]))

        f0 = float(np.log1p(d_out))
        f1 = float(np.log1p(d_in))
        f2 = float(np.log1p(d))
        f3 = float(np.log1p(delta_out))
        f4 = float(np.log1p(delta_in))

        hashed_out = hashed_predicate_profile(rel_deg_out[h], hashed_d_out, d_out)
        hashed_in = hashed_predicate_profile(rel_deg_in[h], hashed_d_in, d_in)

        out_neighs = out_neighbors.get(h, set())
        if len(out_neighs) > 0:
            out_nb_degs = [deg_total.get(str(v), 0) for v in out_neighs]
            out_mean_deg = float(np.mean(out_nb_degs))
            out_max_deg = float(np.max(out_nb_degs))
        else:
            out_mean_deg = 0.0
            out_max_deg = 0.0

        in_neighs = in_neighbors.get(h, set())
        if len(in_neighs) > 0:
            in_nb_degs = [deg_total.get(str(v), 0) for v in in_neighs]
            in_mean_deg = float(np.mean(in_nb_degs))
            in_max_deg = float(np.max(in_nb_degs))
        else:
            in_mean_deg = 0.0
            in_max_deg = 0.0

        if add_cycle_proxy:
            cycle_proxy = cycle_proxy_light_1hop(
                h=h,
                out_neighbors=out_neighbors,
                in_neighbors=in_neighbors,
                neighbors_und=neighbors_und,
                rng=rng,
                sample_size=cycle_sample_size,
            )
        else:
            cycle_proxy = 0.0

        feat = (
            [d_tot, rel_div_tot, avg_nb_deg]
            + rel_feats_total
            + [f0, f1, f2, f3, f4]
            + hashed_out
            + hashed_in
            + [out_mean_deg, out_max_deg, in_mean_deg, in_max_deg]
            + [cycle_proxy]
        )
        X_rows.append(feat)

    X = np.asarray(X_rows, dtype=float)
    print(f"    Feature matrix shape: {X.shape} = (num_heads, num_features)")
    return X


# ---------------------------------------------------------
# kNN structural context (optional) + export similar nodes (optional)
# ---------------------------------------------------------
def add_knn_structural_context(
    X_train_base,
    X_test_base,
    heads_train,
    heads_test,
    neighbors_und,
    ks,
    save_knn_nodes=False,
    knn_nodes_path=None,
):
    if NearestNeighbors is None:
        raise RuntimeError("NearestNeighbors not available (scikit-learn missing neighbors module).")

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    if len(ks) == 0:
        return X_train_base, X_test_base, [], None

    Kmax = max(ks)

    X_all = np.vstack([X_train_base, X_test_base])
    n_train = X_train_base.shape[0]

    sc = StandardScaler()
    Z = sc.fit_transform(X_all)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z = Z / norms

    nn = NearestNeighbors(n_neighbors=Kmax + 1, metric="cosine")
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z, return_distance=True)
    sims = 1.0 - dists

    knn_feats = []
    knn_dump = []

    for i in range(Z.shape[0]):
        nbrs = idxs[i, 1:]
        nbr_sims = sims[i, 1:]

        h_id = str(heads_train[i]) if i < n_train else str(heads_test[i - n_train])
        A_h = neighbors_und.get(h_id, set())

        if save_knn_nodes:
            dump_pairs = []
            for j, s in zip(nbrs[:Kmax], nbr_sims[:Kmax]):
                if j < n_train:
                    u = str(heads_train[j])
                else:
                    u = str(heads_test[j - n_train])
                dump_pairs.append(f"{u}:{float(s):.6f}")
            knn_dump.append((h_id, "|".join(dump_pairs)))

        row = []
        for k in ks:
            topk = nbrs[:k]
            topk_sims = nbr_sims[:k]
            mean_sim = float(np.mean(topk_sims)) if len(topk_sims) > 0 else 0.0

            edge_cnt = 0
            for j in topk:
                if j < n_train:
                    u = str(heads_train[j])
                else:
                    u = str(heads_test[j - n_train])
                if u in A_h:
                    edge_cnt += 1

            row.extend([mean_sim, float(edge_cnt)])

        knn_feats.append(row)

    knn_feats = np.array(knn_feats, dtype=float)
    X_train_out = np.hstack([X_train_base, knn_feats[:n_train]])
    X_test_out = np.hstack([X_test_base, knn_feats[n_train:]])

    feat_names = []
    for k in ks:
        feat_names += [f"knn_mean_sim@{k}", f"knn_edge_count@{k}"]

    saved_path = None
    if save_knn_nodes and knn_nodes_path is not None:
        df_knn = pd.DataFrame(knn_dump, columns=["head_id", "topK_similar_nodes"])
        df_knn.to_csv(knn_nodes_path, sep="\t", index=False)
        saved_path = str(knn_nodes_path)

    return X_train_out, X_test_out, feat_names, saved_path


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--public-path", type=str,
                    default="/home/harrouch/Nell/defense/out_riskmix/graph_defended.tsv")
    ap.add_argument("--sens-path", type=str,
                    default="/home/harrouch/Nell/processed/sensitive/concept:proxyfor.tsv")
    ap.add_argument("--public-has-header", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-train-fraction", type=float, default=0.2)
    ap.add_argument("--train-neg-sample", type=int, default=5000)
    ap.add_argument("--test-neg-sample", type=int, default=5000)

    ap.add_argument("--outdir", type=str, default="/home/harrouch/Nell/results/defense/attack1_s3")
    ap.add_argument("--tag", type=str, default="")

    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1e-4)

    # hard negatives
    ap.add_argument("--hardneg-mode", type=str, default="median_ge",
                    choices=["median_ge", "median_band", "none", "bruteforce_all"])
    ap.add_argument("--hardneg-band-alpha", type=float, default=10.0)

    # features knobs
    ap.add_argument("--hashed-d-out", type=int, default=64)
    ap.add_argument("--hashed-d-in", type=int, default=64)

    # optional cycle proxy
    ap.add_argument("--add-cycle-proxy", action="store_true")
    ap.add_argument("--cycle-sample-size", type=int, default=50)

    # optional kNN
    ap.add_argument("--add-knn", action="store_true")
    ap.add_argument("--knn-ks", type=str, default="5,10")
    ap.add_argument("--save-knn-nodes", action="store_true")

    args = ap.parse_args()

    public_path = Path(args.public_path)
    sens_path = Path(args.sens_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    split_tag = f"{int(args.pos_train_fraction*100)}pct"
    scores_dir = outdir / f"scores_{split_tag}"
    metrics_dir = outdir / f"metrics_{split_tag}"
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scores_path = scores_dir / "head_existence_scores.tsv"
    metrics_path = metrics_dir / "head_existence_mlp.json"
    knn_nodes_path = scores_dir / "knn_similar_nodes.tsv"

    # Load public
    neighbors_und, out_neighbors, in_neighbors, rel_deg_total, rel_deg_out, rel_deg_in, rel_ids = load_public_graph(
        public_path=public_path,
        has_header=bool(args.public_has_header),
    )
    deg_total = {node: len(neighs) for node, neighs in neighbors_und.items()}
    deg_out = {node: len(neighs) for node, neighs in out_neighbors.items()}
    deg_in = {node: len(neighs) for node, neighs in in_neighbors.items()}

    # Load sensitive heads
    pos_heads_set, sens_rel = load_sensitive_heads(sens_path)
    pos_heads_all = np.array(sorted(pos_heads_set), dtype=object)

    all_heads = np.array(sorted(neighbors_und.keys()), dtype=object)
    pos_set = set(pos_heads_all.tolist())
    true_negs_all = np.array([h for h in all_heads if h not in pos_set], dtype=object)

    num_pos_total = len(pos_heads_all)
    num_total = len(all_heads)
    print(f"[+] PUBLIC NODES: total={num_total:,}, pos={num_pos_total:,}, neg={len(true_negs_all):,}")
    if num_pos_total < 2:
        raise RuntimeError("Not enough positives to split train/test (need at least 2).")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(num_pos_total)
    split_idx = int(args.pos_train_fraction * num_pos_total)
    split_idx = max(1, min(split_idx, num_pos_total - 1))

    train_pos = pos_heads_all[perm[:split_idx]]
    test_pos = pos_heads_all[perm[split_idx:]]

    print(f"[+] Positives split: train_pos(seeds)={len(train_pos):,}, test_pos(hidden)={len(test_pos):,}")
    print(f"    Train fraction = {len(train_pos)/num_pos_total:.3f}")

    # TRAIN negatives from unlabeled pool (non-seeds) (may include hidden positives)
    train_pos_set = set(train_pos.tolist())
    train_unlabeled_pool = np.array([h for h in all_heads if h not in train_pos_set], dtype=object)
    print(f"[+] Unlabeled pool for TRAIN (non-seeds): {len(train_unlabeled_pool):,} heads")

    train_neg, med_train, pool_train = hardneg_select(
        mode=args.hardneg_mode,
        neg_candidates=train_unlabeled_pool,
        degrees_total=deg_total,
        ref_pos=train_pos,
        num_sample=args.train_neg_sample,
        rng=rng,
        band_alpha=args.hardneg_band_alpha,
    )
    if med_train is not None:
        print(f"[+] HardNeg train: mode={args.hardneg_mode} median={med_train:.2f} pool={pool_train:,}")

    # TEST negatives from TRUE negatives pool
    rng_test = np.random.default_rng(args.seed + 1)
    test_neg, med_test, pool_test = hardneg_select(
        mode=args.hardneg_mode,
        neg_candidates=true_negs_all,
        degrees_total=deg_total,
        ref_pos=train_pos,
        num_sample=args.test_neg_sample,
        rng=rng_test,
        band_alpha=args.hardneg_band_alpha,
    )
    if med_test is not None:
        print(f"[+] HardNeg test:  mode={args.hardneg_mode} median={med_test:.2f} pool={pool_test:,}")

    heads_train = np.concatenate([train_pos, train_neg])
    y_train = np.concatenate([np.ones(len(train_pos), dtype=int), np.zeros(len(train_neg), dtype=int)])

    heads_test = np.concatenate([test_pos, test_neg])
    y_test = np.concatenate([np.ones(len(test_pos), dtype=int), np.zeros(len(test_neg), dtype=int)])

    print(f"[+] TRAIN set: {len(heads_train):,} heads ({int(y_train.sum()):,} pos, {len(heads_train)-int(y_train.sum()):,} neg)")
    print(f"[+] TEST  set: {len(heads_test):,} heads ({int(y_test.sum()):,} pos, {len(heads_test)-int(y_test.sum()):,} neg)")

    X_train = build_head_features(
        head_ids=heads_train,
        neighbors_und=neighbors_und,
        out_neighbors=out_neighbors,
        in_neighbors=in_neighbors,
        rel_deg_total=rel_deg_total,
        rel_deg_out=rel_deg_out,
        rel_deg_in=rel_deg_in,
        rel_ids=rel_ids,
        deg_total=deg_total,
        deg_out=deg_out,
        deg_in=deg_in,
        hashed_d_out=args.hashed_d_out,
        hashed_d_in=args.hashed_d_in,
        add_cycle_proxy=bool(args.add_cycle_proxy),
        cycle_sample_size=int(args.cycle_sample_size),
        rng_seed=args.seed,
    )
    X_test = build_head_features(
        head_ids=heads_test,
        neighbors_und=neighbors_und,
        out_neighbors=out_neighbors,
        in_neighbors=in_neighbors,
        rel_deg_total=rel_deg_total,
        rel_deg_out=rel_deg_out,
        rel_deg_in=rel_deg_in,
        rel_ids=rel_ids,
        deg_total=deg_total,
        deg_out=deg_out,
        deg_in=deg_in,
        hashed_d_out=args.hashed_d_out,
        hashed_d_in=args.hashed_d_in,
        add_cycle_proxy=bool(args.add_cycle_proxy),
        cycle_sample_size=int(args.cycle_sample_size),
        rng_seed=args.seed,
    )

    # optional kNN
    knn_feat_names = []
    saved_knn_nodes_path = None
    if args.add_knn:
        ks = [int(x) for x in args.knn_ks.split(",") if x.strip().isdigit()]
        X_train, X_test, knn_feat_names, saved_knn_nodes_path = add_knn_structural_context(
            X_train_base=X_train,
            X_test_base=X_test,
            heads_train=heads_train,
            heads_test=heads_test,
            neighbors_und=neighbors_und,
            ks=ks,
            save_knn_nodes=bool(args.save_knn_nodes),
            knn_nodes_path=knn_nodes_path if args.save_knn_nodes else None,
        )
        print(f"[+] Added kNN features: {knn_feat_names}")
        if saved_knn_nodes_path is not None:
            print(f"[+] Saved kNN similar nodes -> {saved_knn_nodes_path}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    hidden_sizes = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    print(f"[+] Training MLPClassifier hidden={hidden_sizes} max_iter={args.max_iter}")

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=False,
    )
    mlp.fit(X_train_sc, y_train)

    print("[+] Evaluating on TEST")
    y_scores = mlp.predict_proba(X_test_sc)[:, 1]

    pr_auc_val = float(average_precision_score(y_test, y_scores))
    try:
        roc_auc_val = float(roc_auc_score(y_test, y_scores))
    except Exception:
        roc_auc_val = None

    print(f"    PR-AUC = {pr_auc_val:.4f}")
    if roc_auc_val is not None:
        print(f"    ROC-AUC = {roc_auc_val:.4f}")

    df_scores = pd.DataFrame({"head_id": heads_test, "label": y_test, "score": y_scores})
    df_scores.to_csv(scores_path, sep="\t", index=False)
    print(f"[+] Saved scores -> {scores_path}")

    metrics = {
        "tag": args.tag,
        "sensitive_relation": sens_rel,
        "attack_type": "attack1_head_existence_mlp_structure_partial_supervision",
        "public_path_used": str(public_path),
        "sens_path_used": str(sens_path),

        "num_nodes_total_public": int(len(all_heads)),
        "num_pos_total": int(len(pos_heads_all)),
        "num_neg_total": int(len(true_negs_all)),

        "pos_train_fraction": float(len(train_pos) / len(pos_heads_all)),
        "num_train_pos": int(len(train_pos)),
        "num_test_pos": int(len(test_pos)),
        "num_train_neg": int(len(train_neg)),
        "num_test_neg": int(len(test_neg)),

        "hardneg_mode": str(args.hardneg_mode),
        "hardneg_band_alpha": float(args.hardneg_band_alpha),
        "hardneg_median_degree_trainpos": float(med_train) if med_train is not None else None,

        "num_features": int(X_train.shape[1]),
        "mlp_hidden_sizes": list(hidden_sizes),
        "mlp_alpha": float(args.alpha),
        "mlp_max_iter": int(args.max_iter),

        "pr_auc_test": pr_auc_val,
        "roc_auc_test": roc_auc_val,
        "random_seed": int(args.seed),

        "train_neg_sample": int(args.train_neg_sample),
        "test_neg_sample": int(args.test_neg_sample),

        "hashed_d_out": int(args.hashed_d_out),
        "hashed_d_in": int(args.hashed_d_in),
        "add_cycle_proxy": bool(args.add_cycle_proxy),
        "cycle_sample_size": int(args.cycle_sample_size),

        "add_knn": bool(args.add_knn),
        "knn_ks": str(args.knn_ks),
        "knn_feature_names": knn_feat_names,
        "knn_nodes_saved_path": saved_knn_nodes_path,

        "public_has_header": bool(args.public_has_header),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Saved metrics -> {metrics_path}")
    print("[+] Done.")


if __name__ == "__main__":
    main()

