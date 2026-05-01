
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ----------------------------
# utils
# ----------------------------
def as_id(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def load_tsv_3cols(path: Path, has_header: bool):
    df = pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        usecols=[0, 1, 2],
        names=["head_id", "rel_id", "tail_id"] if not has_header else None,
        dtype=str,
        low_memory=False,
    ).dropna()
    df = df.rename(columns={df.columns[0]: "head_id", df.columns[1]: "rel_id", df.columns[2]: "tail_id"})
    df["head_id"] = df["head_id"].map(as_id)
    df["rel_id"] = df["rel_id"].map(as_id)
    df["tail_id"] = df["tail_id"].map(as_id)
    df = df.dropna()
    return df


# ----------------------------
# Public graph
# ----------------------------
def load_public_graph(public_path: Path, public_has_header: bool):
    print(f"[+] Loading public graph: {public_path}")
    df = load_tsv_3cols(public_path, public_has_header)
    print(f"    Public triples loaded: {len(df):,}")

    neighbors = defaultdict(set)  # undirected
    rel_degrees = defaultdict(lambda: defaultdict(int))

    for h, r, t in df.itertuples(index=False, name=None):
        neighbors[h].add(t)
        neighbors[t].add(h)
        rel_degrees[h][r] += 1
        rel_degrees[t][r] += 1

    rel_ids = sorted(df["rel_id"].unique().tolist())
    degrees = {n: len(v) for n, v in neighbors.items()}
    print(f"[+] Nodes={len(neighbors):,} | Relations={len(rel_ids):,}")
    return neighbors, rel_degrees, rel_ids, degrees


# ----------------------------
# Sensitive multi-tail GT
# ----------------------------
def load_sensitive_head_to_tailset(
    sens_path: Path,
    sens_has_header: bool,
    relation_filter: str | None,
    head_prefix: str | None,
):
    print(f"[+] Loading sensitive triples: {sens_path}")
    df = load_tsv_3cols(sens_path, sens_has_header)

    if head_prefix:
        df = df[df["head_id"].str.startswith(head_prefix, na=False)]
    if relation_filter:
        df = df[df["rel_id"] == str(relation_filter)]

    print(f"    Sensitive triples kept: {len(df):,} | unique_heads={df['head_id'].nunique():,}")
    if df.empty:
        raise RuntimeError("No sensitive triples after filtering (relation/prefix).")

    head2tails = defaultdict(set)
    for h, t in zip(df["head_id"].tolist(), df["tail_id"].tolist()):
        head2tails[h].add(t)

    heads = np.array(sorted(head2tails.keys()), dtype=object)
    return head2tails, heads


# ----------------------------
# Attack1 scores loader (head or tail)
# ----------------------------
def load_attack1_scores(scores_path: Path, id_col: str, score_col: str):
    df = pd.read_csv(scores_path, sep="\t", dtype=str).dropna()
    if id_col not in df.columns or score_col not in df.columns:
        raise RuntimeError(
            f"Attack1 file must contain columns '{id_col}' and '{score_col}'. Found: {list(df.columns)}"
        )
    df = df[[id_col, score_col]].dropna()
    df[id_col] = df[id_col].map(as_id)
    df[score_col] = df[score_col].astype(float)
    return dict(zip(df[id_col].tolist(), df[score_col].tolist()))


# ----------------------------
# Light proxies for node features
# ----------------------------
def layered_proxies_light(node_id, neighbors, sample_size=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    N1 = neighbors.get(node_id, set())
    if not N1:
        return 0.0, 0.0, 0.0

    N2 = set()
    for u in N1:
        N2.update(neighbors.get(u, set()))
    N2.discard(node_id)
    N2.difference_update(N1)
    n2_size = float(len(N2))

    N1_list = list(N1)
    if len(N1_list) > sample_size:
        N1_s = rng.choice(np.array(N1_list, dtype=object), size=sample_size, replace=False).tolist()
    else:
        N1_s = N1_list
    N1_s_set = set(N1_s)

    I1_cnt = 0
    for u in N1_s:
        I1_cnt += len(N1_s_set.intersection(neighbors.get(u, set())))
    I1 = float(I1_cnt) / 2.0

    E1_cnt = 0
    for u in N1_s:
        E1_cnt += len(N2.intersection(neighbors.get(u, set())))
    E1 = float(E1_cnt)

    return n2_size, E1, I1


def build_node_features(node_ids, neighbors, rel_degrees, rel_ids, degrees,
                        feature_set: str, tri_sample_size: int, rng):
    """
    feature_set:
      - local
      - proxies
      - knn   (base features = proxies; kNN context will be added later to HEADS only)
    """
    X_rows = []
    kept_ids = []

    base_set = "proxies" if feature_set == "knn" else feature_set

    for n in node_ids:
        n = as_id(n)
        if n is None:
            continue

        deg = degrees.get(n, 0)
        rel_dict = rel_degrees[n]
        rel_div = len(rel_dict)

        neighs = neighbors.get(n, set())
        avg_nb_deg = float(np.mean([degrees.get(v, 0) for v in neighs])) if neighs else 0.0

        rel_counts = [float(rel_dict.get(r, 0)) for r in rel_ids]

        if base_set == "local":
            feat = [float(deg), float(rel_div), float(avg_nb_deg)] + rel_counts
        elif base_set == "proxies":
            n2_size, e1, i1 = layered_proxies_light(n, neighbors, sample_size=tri_sample_size, rng=rng)
            feat = [float(deg), float(rel_div), float(avg_nb_deg),
                    float(n2_size), float(e1), float(i1)] + rel_counts
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")

        X_rows.append(feat)
        kept_ids.append(n)

    return np.asarray(X_rows, dtype=np.float32), np.array(kept_ids, dtype=object)


# ----------------------------
# kNN structural context (HEADS ONLY)
# ----------------------------
def add_knn_context_heads(Xh_sc: np.ndarray, head_ids: np.ndarray, neighbors_und: dict, k: int):
    """
    On standardized head features:
      - L2 normalize
      - cosine kNN
      - add [mean_sim, edge_cnt] as extra features
    """
    k = int(k)
    if k <= 0 or Xh_sc.shape[0] < 2:
        return Xh_sc

    Z = Xh_sc.astype(np.float32, copy=True)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Z = Z / norms

    nn = NearestNeighbors(n_neighbors=min(k + 1, Z.shape[0]), metric="cosine")
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z, return_distance=True)
    sims = 1.0 - dists

    extra = np.zeros((Z.shape[0], 2), dtype=np.float32)

    for i in range(Z.shape[0]):
        nbrs = idxs[i, 1:]
        nbr_sims = sims[i, 1:]
        mean_sim = float(np.mean(nbr_sims)) if len(nbr_sims) else 0.0

        h = as_id(head_ids[i])
        A_h = neighbors_und.get(h, set())

        edge_cnt = 0
        for j in nbrs:
            u = as_id(head_ids[int(j)])
            if u in A_h:
                edge_cnt += 1

        extra[i, 0] = mean_sim
        extra[i, 1] = float(edge_cnt)

    return np.hstack([Xh_sc, extra]).astype(np.float32)


# ----------------------------
# Pairwise structural features
# ----------------------------
def pair_struct_features(h, t, neighbors, degrees):
    Nh = neighbors.get(h, set())
    Nt = neighbors.get(t, set())

    is1 = 1.0 if t in Nh else 0.0

    is2 = 0.0
    if is1 == 0.0 and Nh:
        if len(Nh) <= 2000:
            for u in Nh:
                if t in neighbors.get(u, set()):
                    is2 = 1.0
                    break
        else:
            if Nt and (len(Nh.intersection(Nt)) > 0):
                is2 = 1.0

    if Nh and Nt:
        inter = Nh.intersection(Nt)
        cn = float(len(inter))
        denom = float(len(Nh) + len(Nt) - len(inter))
        jac = cn / denom if denom > 0 else 0.0
    else:
        cn = 0.0
        jac = 0.0

    pa = float(degrees.get(h, 0) * degrees.get(t, 0))
    pa = math.log1p(pa)

    return np.array([is1, is2, cn, jac, pa], dtype=np.float32)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--public-path", required=True, type=str)
    ap.add_argument("--public-has-header", action="store_true", default=False)

    ap.add_argument("--sens-path", required=True, type=str)
    ap.add_argument("--sens-has-header", action="store_true", default=False)

    ap.add_argument("--relation-name", type=str, default="relation")  # naming
    ap.add_argument("--relation-filter", type=str, default="")        # filter in sensitive
    ap.add_argument("--head-prefix", type=str, default="")

    # Attack1-head
    ap.add_argument("--attack1-head-scores", required=True, type=str)
    ap.add_argument("--a1h-id-col", type=str, default="head_id")
    ap.add_argument("--a1h-score-col", type=str, default="score")
    ap.add_argument("--a1h-thr", type=float, default=0.5)

    # Attack1-tail
    ap.add_argument("--attack1-tail-scores", required=True, type=str)
    ap.add_argument("--a1t-id-col", type=str, default="tail_id")
    ap.add_argument("--a1t-score-col", type=str, default="score")
    ap.add_argument("--a1t-thr", type=float, default=0.5)

    # split
    ap.add_argument("--seed-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # Features + model
    ap.add_argument("--feature-set", choices=["local", "proxies", "knn"], default="local")
    ap.add_argument("--tri-sample-size", type=int, default=50)
    ap.add_argument("--knn-k", type=int, default=50)

    ap.add_argument("--neg-per-pos", type=int, default=20)
    ap.add_argument("--include-pair-feats", action="store_true", default=True)
    ap.add_argument("--no-include-pair-feats", dest="include_pair_feats", action="store_false")

    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=1e-4)

    # scoring
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--score-cap-per-head", type=int, default=0)

    ap.add_argument("--pool-include-seed-tails", action="store_true", default=True)
    ap.add_argument("--outdir", type=str, default="results_attack2_from_a1")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    head_prefix = args.head_prefix.strip() or None
    rel_filter = args.relation_filter.strip() or None

    run_name = (
        f"attack2_fromA1_{args.relation_name}"
        f"_A1H{args.a1h_thr:g}_A1T{args.a1t_thr:g}"
        f"_{args.feature_set}_seed{args.seed_frac:g}"
    )
    base = Path(args.outdir) / run_name
    scores_dir = base / "scores"
    metrics_dir = base / "metrics"
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scores_out = scores_dir / f"{args.relation_name}_attack2_pair_scores.tsv"
    metrics_out = metrics_dir / f"{args.relation_name}_attack2_pair_metrics.json"

    # public
    neighbors, rel_degrees, rel_ids, degrees = load_public_graph(Path(args.public_path), args.public_has_header)

    # sensitive
    head2gt, gt_heads = load_sensitive_head_to_tailset(
        sens_path=Path(args.sens_path),
        sens_has_header=args.sens_has_header,
        relation_filter=rel_filter,
        head_prefix=head_prefix,
    )

    # A1 head -> candidate heads
    a1h = load_attack1_scores(Path(args.attack1_head_scores), args.a1h_id_col, args.a1h_score_col)
    cand_heads = np.array(sorted([h for h, s in a1h.items() if float(s) >= float(args.a1h_thr)]), dtype=object)
    cand_heads = cand_heads
    print(f"[+] Candidate heads from Attack1-head: {len(cand_heads):,} (thr={args.a1h_thr})")
    if len(cand_heads) == 0:
        raise RuntimeError("No candidate heads after Attack1-head threshold.")

    # A1 tail -> tail pool
    a1t = load_attack1_scores(Path(args.attack1_tail_scores), args.a1t_id_col, args.a1t_score_col)
    tail_pool = np.array(sorted([t for t, s in a1t.items() if float(s) >= float(args.a1t_thr)]), dtype=object)
    tail_pool = tail_pool
    print(f"[+] Tail pool from Attack1-tail: {len(tail_pool):,} (thr={args.a1t_thr})")
    if len(tail_pool) == 0:
        raise RuntimeError("Tail pool empty after Attack1-tail threshold.")

    # focus on GT heads that are inside candidates
    cand_set = set(cand_heads.tolist())
    gt_heads_in_cand = np.array([h for h in gt_heads if h in cand_set], dtype=object)
    print(f"[+] GT heads inside candidate heads: {len(gt_heads_in_cand):,} / {len(gt_heads):,}")
    if len(gt_heads_in_cand) < 10:
        raise RuntimeError("Too few GT heads intersect candidate heads; lower --a1h-thr.")

    # split seeds vs hidden
    idx = np.arange(len(gt_heads_in_cand))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=max(0.0, min(1.0, 1.0 - float(args.seed_frac))),
        random_state=args.seed,
        shuffle=True,
    )
    seed_heads = gt_heads_in_cand[train_idx].astype(object)
    hidden_heads = gt_heads_in_cand[test_idx].astype(object)
    print(f"[+] Split GT heads: seeds={len(seed_heads):,} hidden={len(hidden_heads):,} seed_frac={args.seed_frac}")

    # final pool include seed tails
    if args.pool_include_seed_tails:
        seed_tails = set()
        for h in seed_heads.tolist():
            seed_tails.update(head2gt.get(h, set()))
        seed_tails = [t for t in seed_tails if t in neighbors]
        tail_pool_final = np.array(sorted(set(tail_pool.tolist()).union(seed_tails)), dtype=object)
    else:
        tail_pool_final = tail_pool.copy()
    print(f"[+] Tail pool final size={len(tail_pool_final):,} (include_seed_tails={args.pool_include_seed_tails})")

    # node features
    Xh_all, heads_kept = build_node_features(
        node_ids=cand_heads,
        neighbors=neighbors,
        rel_degrees=rel_degrees,
        rel_ids=rel_ids,
        degrees=degrees,
        feature_set=args.feature_set,
        tri_sample_size=args.tri_sample_size,
        rng=rng,
    )
    head2row = {as_id(h): i for i, h in enumerate(heads_kept.tolist())}

    Xt_pool, tails_kept = build_node_features(
        node_ids=tail_pool_final,
        neighbors=neighbors,
        rel_degrees=rel_degrees,
        rel_ids=rel_ids,
        degrees=degrees,
        feature_set=args.feature_set,
        tri_sample_size=args.tri_sample_size,
        rng=rng,
    )
    tail2row = {as_id(t): i for i, t in enumerate(tails_kept.tolist())}

    # scale
    head_scaler = StandardScaler()
    Xh_all_sc = head_scaler.fit_transform(Xh_all).astype(np.float32)

    # add kNN context to HEADS ONLY
    if args.feature_set == "knn":
        Xh_all_sc = add_knn_context_heads(
            Xh_sc=Xh_all_sc,
            head_ids=heads_kept,
            neighbors_und=neighbors,
            k=int(args.knn_k),
        )
        print(f"[+] Added head kNN context: knn_k={args.knn_k} head_dim={Xh_all_sc.shape[1]}")

    tail_scaler = StandardScaler()
    Xt_pool_sc = tail_scaler.fit_transform(Xt_pool).astype(np.float32)

    # build pairwise train set
    print("[+] Building pairwise TRAIN set from seeds ...")
    X_pairs, y_pairs = [], []
    pool_arr = tails_kept.copy()

    for h in seed_heads.tolist():
        h = as_id(h)
        if h not in head2row:
            continue

        gt_tails = [t for t in head2gt.get(h, set()) if t in tail2row]
        if not gt_tails:
            continue

        h_feat = Xh_all_sc[head2row[h]]

        for true_t in gt_tails:
            t_feat = Xt_pool_sc[tail2row[true_t]]

            x = np.hstack([h_feat, t_feat])
            if args.include_pair_feats:
                x = np.hstack([x, pair_struct_features(h, true_t, neighbors, degrees)])
            X_pairs.append(x)
            y_pairs.append(1)

            kneg = int(args.neg_per_pos)
            if kneg <= 0:
                continue

            tries, negs = 0, []
            gt_set = set(gt_tails)
            while len(negs) < kneg and tries < kneg * 50:
                cand = as_id(pool_arr[int(rng.integers(0, len(pool_arr)))])
                if cand not in gt_set:
                    negs.append(cand)
                tries += 1

            for neg_t in negs:
                t_feat_n = Xt_pool_sc[tail2row[neg_t]]
                xneg = np.hstack([h_feat, t_feat_n])
                if args.include_pair_feats:
                    xneg = np.hstack([xneg, pair_struct_features(h, neg_t, neighbors, degrees)])
                X_pairs.append(xneg)
                y_pairs.append(0)

    if len(y_pairs) < 200:
        raise RuntimeError("Too few train pairs (increase seeds / lower thresholds / increase pool).")

    X_pairs = np.asarray(X_pairs, dtype=np.float32)
    y_pairs = np.asarray(y_pairs, dtype=np.int64)
    print(f"    Train pairs: {len(y_pairs):,} (pos={int(y_pairs.sum()):,}, neg={int((y_pairs==0).sum()):,})")

    pair_scaler = StandardScaler()
    X_pairs_sc = pair_scaler.fit_transform(X_pairs).astype(np.float32)

    hidden_sizes = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    print(f"[+] Training binary MLP scorer hidden={hidden_sizes} max_iter={args.max_iter}")

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        alpha=float(args.alpha),
        max_iter=int(args.max_iter),
        random_state=int(args.seed),
        verbose=False,
    )
    clf.fit(X_pairs_sc, y_pairs)

    # score candidate heads over pool
    print("[+] Scoring candidate heads over tail pool ...")
    out_rows = []
    topk = int(args.topk)

    for h in cand_heads.tolist():
        h = as_id(h)
        if h not in head2row:
            continue

        cand_tails = tails_kept
        if args.score_cap_per_head and int(args.score_cap_per_head) > 0 and len(cand_tails) > int(args.score_cap_per_head):
            cand_tails = rng.choice(cand_tails, size=int(args.score_cap_per_head), replace=False)

        h_feat = Xh_all_sc[head2row[h]]

        idxs = np.array([tail2row[as_id(t)] for t in cand_tails.tolist()], dtype=np.int64)
        t_feats = Xt_pool_sc[idxs]

        H = np.repeat(h_feat.reshape(1, -1), repeats=len(cand_tails), axis=0)
        X_ht = np.hstack([H, t_feats])

        if args.include_pair_feats:
            P = np.zeros((len(cand_tails), 5), dtype=np.float32)
            for i, t in enumerate(cand_tails.tolist()):
                P[i, :] = pair_struct_features(h, as_id(t), neighbors, degrees)
            X_ht = np.hstack([X_ht, P])

        X_ht_sc = pair_scaler.transform(X_ht).astype(np.float32)
        scores = clf.predict_proba(X_ht_sc)[:, 1].astype(np.float32)

        order = np.argsort(-scores)
        if topk > 0:
            order = order[:topk]

        for rank_i, j in enumerate(order, start=1):
            out_rows.append({
                "head_id": h,
                "tail_id": as_id(cand_tails[int(j)]),
                "score": float(scores[int(j)]),
                "rank": int(rank_i),
            })

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(scores_out, sep="\t", index=False)
    print(f"[+] Saved scores -> {scores_out} (rows={len(df_out):,})")

    metrics = {
        "relation_name": args.relation_name,
        "relation_filter": rel_filter,
        "attack_type": "attack2_pairwise_from_attack1_head_and_tail",
        "paths": {
            "public": str(Path(args.public_path)),
            "sensitive": str(Path(args.sens_path)),
            "attack1_head_scores": str(Path(args.attack1_head_scores)),
            "attack1_tail_scores": str(Path(args.attack1_tail_scores)),
        },
        "thresholds": {"a1h_thr": float(args.a1h_thr), "a1t_thr": float(args.a1t_thr)},
        "sizes": {
            "candidate_heads": int(len(cand_heads)),
            "gt_heads_total": int(len(gt_heads)),
            "gt_heads_in_candidates": int(len(gt_heads_in_cand)),
            "seed_heads": int(len(seed_heads)),
            "hidden_heads": int(len(hidden_heads)),
            "tail_pool_a1": int(len(tail_pool)),
            "tail_pool_final": int(len(tail_pool_final)),
        },
        "training": {
            "seed_frac": float(args.seed_frac),
            "neg_per_pos": int(args.neg_per_pos),
            "include_pair_feats": bool(args.include_pair_feats),
            "mlp_hidden": list(hidden_sizes),
            "mlp_alpha": float(args.alpha),
            "mlp_max_iter": int(args.max_iter),
        },
        "features": {
            "feature_set": args.feature_set,
            "tri_sample_size": int(args.tri_sample_size),
            "knn_k": int(args.knn_k) if args.feature_set == "knn" else None,
            "num_rel_features": int(len(rel_ids)),
        },
        "scoring": {"topk_saved": int(args.topk), "score_cap_per_head": int(args.score_cap_per_head)},
    }

    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved metrics -> {metrics_out}")
    print("[+] Done.")


if __name__ == "__main__":
    main()
