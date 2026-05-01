#!/usr/bin/env python3
import argparse, json, math, random, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import unquote

import numpy as np
import pandas as pd

# PR-AUC
from sklearn.metrics import precision_recall_curve, auc

# Optional FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


# -----------------------
# Utils / Normalization
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def norm_ent(x: str) -> str:
    if x is None:
        return ""
    x = x.strip()
    if not x:
        return ""
    if x.startswith("<") and x.endswith(">"):
        x = x[1:-1].strip()
    x = unquote(x)
    if "://" in x:
        x = x.rsplit("/", 1)[-1]
    return x.strip()

def hidx(s: str, dim: int, salt: str) -> int:
    b = (salt + "||" + s).encode("utf-8", errors="ignore")
    hv = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(hv, "little") % dim

def read_public_triples(path: Path):
    # head \t rel \t tail
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            h = norm_ent(parts[0])
            r = parts[1].strip()
            t = norm_ent(parts[2])
            if not h or not t or not r:
                continue
            yield h, r, t

def rel_from_filename(fn: str) -> str:
    x = fn
    if x.endswith(".tsv"):
        x = x[:-4]
    return x

def read_sensitive_flexible(path: Path, rel_name: str):
    """
    Supports:
      - 3 cols: h \t rel \t t
      - 2 cols: h \t t
    Normalizes head and tail.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                h, t = parts[0], parts[2]
            elif len(parts) == 2:
                h, t = parts[0], parts[1]
            else:
                parts = line.split()
                if len(parts) >= 3:
                    h, t = parts[0], parts[2]
                elif len(parts) == 2:
                    h, t = parts[0], parts[1]
                else:
                    continue
            h = norm_ent(h)
            t = norm_ent(t)
            if not h or not t:
                continue
            yield h, rel_name, t


# -----------------------
# PR-AUC helper
# -----------------------
def pr_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray):
    """
    Candidate-based PR-AUC. Returns NaN if undefined (no positives or no negatives).
    """
    if y_true.size == 0:
        return np.nan
    pos = int(y_true.sum())
    if pos == 0 or pos == y_true.size:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


# -----------------------
# Ranking metrics (MRR / Hits@K) over candidate tails per head
# -----------------------
def ranking_metrics_from_votes(votes_by_head: dict, true_tails_by_head: dict, Ks=(10,)):
    ranks = []
    hits = {k: 0 for k in Ks}
    n_eval = 0

    for h, votes in votes_by_head.items():
        T = true_tails_by_head.get(h, set())
        if not T:
            continue
        n_eval += 1

        items = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        cand_rank = {t: (i + 1) for i, (t, _) in enumerate(items)}

        best_rank = math.inf
        for t in T:
            if t in cand_rank:
                best_rank = min(best_rank, cand_rank[t])

        ranks.append(best_rank)
        for k in Ks:
            if best_rank <= k:
                hits[k] += 1

    if n_eval == 0:
        out = {"N_eval_heads": 0, "MRR": np.nan, "MR": np.nan, "MedianRank": np.nan}
        for k in Ks:
            out[f"Hits@{k}"] = np.nan
        return out

    rr = [(1.0 / r) if np.isfinite(r) else 0.0 for r in ranks]
    finite_r = [r for r in ranks if np.isfinite(r)]

    out = {
        "N_eval_heads": int(n_eval),
        "MRR": float(np.mean(rr)),
        "MR": float(np.mean(finite_r)) if finite_r else float("inf"),
        "MedianRank": float(np.median(finite_r)) if finite_r else float("inf"),
    }
    for k in Ks:
        out[f"Hits@{k}"] = float(hits[k] / n_eval)
    return out


# -----------------------
# Cardinality stats
# -----------------------
def relation_cardinality_stats(Hmap: dict):
    if not Hmap:
        return 0.0, 0.0, "EMPTY"

    tails = Counter()
    tails_per_head = []
    for h, ts in Hmap.items():
        tails_per_head.append(len(ts))
        for t in ts:
            tails[t] += 1

    avg_tph = float(np.mean(tails_per_head)) if tails_per_head else 0.0
    avg_hpt = float(np.mean(list(tails.values()))) if tails else 0.0

    left = "1" if avg_tph <= 1.5 else "N"
    right = "1" if avg_hpt <= 1.5 else "N"
    return avg_tph, avg_hpt, f"{left}-{right}"


def build_metrics_table(true_edges_by_rel, pred_edges_by_rel, rels, gt_Hmap_by_rel,
                        pr_auc_by_rel=None, pr_auc_global=np.nan,
                        rank_by_rel=None, rank_global=None):
    rows = []
    tp_tot = fp_tot = fn_tot = 0
    true_tot = 0

    def _get_rank_val(r, key, default=np.nan):
        if isinstance(rank_by_rel, dict) and r in rank_by_rel and key in rank_by_rel[r]:
            return rank_by_rel[r][key]
        return default

    for r in rels:
        T = true_edges_by_rel.get(r, set())
        P = pred_edges_by_rel.get(r, set())
        inter = T & P
        tp = len(inter)
        fp = len(P) - tp
        fn = len(T) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1   = (2 * prec * rec / (prec + rec)) if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else np.nan

        avg_tph, avg_hpt, rtype = relation_cardinality_stats(gt_Hmap_by_rel.get(r, {}))

        rows.append({
            "Relation": r,
            "Type": rtype,
            "AvgTails/Head": round(avg_tph, 3),
            "AvgHeads/Tail": round(avg_hpt, 3),
            "True Edges": len(T),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Recall (%)": (rec*100.0) if not np.isnan(rec) else np.nan,
            "Precision (%)": (prec*100.0) if not np.isnan(prec) else np.nan,
            "F1 (%)": (f1*100.0) if not np.isnan(f1) else np.nan,
            "PR-AUC": (pr_auc_by_rel.get(r, np.nan) if isinstance(pr_auc_by_rel, dict) else np.nan),
            "MRR": _get_rank_val(r, "MRR"),
            "Hits@10": _get_rank_val(r, "Hits@10"),
        })

        tp_tot += tp; fp_tot += fp; fn_tot += fn
        true_tot += len(T)

    prec_g = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else np.nan
    rec_g  = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else np.nan
    f1_g   = (2 * prec_g * rec_g / (prec_g + rec_g)) if (not np.isnan(prec_g) and not np.isnan(rec_g) and (prec_g + rec_g) > 0) else np.nan

    rows.append({
        "Relation": "GLOBAL",
        "Type": "",
        "AvgTails/Head": "",
        "AvgHeads/Tail": "",
        "True Edges": true_tot,
        "TP": tp_tot,
        "FP": fp_tot,
        "FN": fn_tot,
        "Recall (%)": (rec_g*100.0) if not np.isnan(rec_g) else np.nan,
        "Precision (%)": (prec_g*100.0) if not np.isnan(prec_g) else np.nan,
        "F1 (%)": (f1_g*100.0) if not np.isnan(f1_g) else np.nan,
        "PR-AUC": pr_auc_global,
        "MRR": (rank_global.get("MRR") if isinstance(rank_global, dict) else np.nan),
        "Hits@10": (rank_global.get("Hits@10") if isinstance(rank_global, dict) else np.nan),
    })

    return pd.DataFrame(rows)


# -----------------------
# Struct v3 builder (PUBLIC ONLY) for target heads
# -----------------------
def build_struct_v3(public_tsv: Path, heads_focus: set, out_csv: Path,
                    hash_dim_out: int, hash_dim_in: int,
                    max_out_neighbors: int, progress_every: int = 5_000_000):
    print("[StructV3] Pass1: computing global degrees ...")
    deg = Counter()
    n = 0
    for h, r, t in read_public_triples(public_tsv):
        deg[h] += 1
        deg[t] += 1
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"  Pass1 processed {n:,} triples")
    print(f"[StructV3] Pass1 done. entities_with_degree={len(deg):,} triples={n:,}")

    print("[StructV3] Pass2: building head features ...")
    out_deg = Counter()
    in_deg  = Counter()
    out_rels = defaultdict(set)
    in_rels  = defaultdict(set)

    out_hash = defaultdict(lambda: np.zeros(hash_dim_out, dtype=np.float32))
    in_hash  = defaultdict(lambda: np.zeros(hash_dim_in, dtype=np.float32))
    out_neighbors = defaultdict(list)

    n2 = 0
    for h, r, t in read_public_triples(public_tsv):
        if h in heads_focus:
            out_deg[h] += 1
            out_rels[h].add(r)
            out_hash[h][hidx(r, hash_dim_out, "OUTPRED")] += 1.0
            if max_out_neighbors <= 0 or len(out_neighbors[h]) < max_out_neighbors:
                out_neighbors[h].append(t)

        if t in heads_focus:
            in_deg[t] += 1
            in_rels[t].add(r)
            in_hash[t][hidx(r, hash_dim_in, "INPRED")] += 1.0

        n2 += 1
        if progress_every and n2 % progress_every == 0:
            print(f"  Pass2 processed {n2:,} triples")

    rows = []
    heads_sorted = sorted(heads_focus)
    for h in heads_sorted:
        od = int(out_deg[h])
        idg = int(in_deg[h])
        td = od + idg

        nb = out_neighbors.get(h, [])
        if nb:
            nb_deg = [deg.get(x, 0) for x in nb]
            nb_mean = float(np.mean(nb_deg)) if nb_deg else 0.0
            nb_max  = float(np.max(nb_deg)) if nb_deg else 0.0
        else:
            nb_mean = 0.0
            nb_max = 0.0

        base_feats = np.array([
            math.log1p(od),
            math.log1p(idg),
            math.log1p(td),
            math.log1p(len(out_rels[h])),
            math.log1p(len(in_rels[h])),
            nb_mean,
            nb_max
        ], dtype=np.float32)

        oh = out_hash[h].copy()
        ih = in_hash[h].copy()
        if od > 0:  oh /= float(od)
        if idg > 0: ih /= float(idg)

        feats = np.concatenate([base_feats, oh, ih], axis=0)

        row = {"head": h}
        for j, v in enumerate(feats):
            row[f"f{j}"] = float(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"[StructV3] Wrote {out_csv} rows={len(df):,} cols={len(df.columns):,}")
    return df


# -----------------------
# Standardization
# -----------------------
def zfit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd

def zapply(X, mu, sd):
    return (X - mu) / sd


# -----------------------
# kNN backends
# -----------------------
def knn_faiss(X: np.ndarray, k: int, use_gpu: bool = True, gpu_id: int = 0):
    n, d = X.shape
    index = faiss.IndexFlatIP(d)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
            print(f"[kNN] Using FAISS GPU (gpu_id={gpu_id})")
        except Exception:
            print("[kNN] FAISS GPU failed, fallback to CPU")
            index = faiss.IndexFlatIP(d)

    Xc = np.ascontiguousarray(X, dtype=np.float32)
    index.add(Xc)
    D, I = index.search(Xc, k+1)
    return I[:, 1:k+1].astype(np.int32)

def knn_batched_cosine(Xn: np.ndarray, k: int, batch: int = 5000):
    n, d = Xn.shape
    I_out = np.empty((n, k), dtype=np.int32)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        sim = Xn[i0:i1] @ Xn.T
        for i in range(i0, i1):
            sim[i - i0, i] = -1e9
        idx = np.argpartition(-sim, kth=k-1, axis=1)[:, :k]
        sc = np.take_along_axis(sim, idx, axis=1)
        order = np.argsort(-sc, axis=1)
        idx_sorted = np.take_along_axis(idx, order, axis=1)
        I_out[i0:i1] = idx_sorted.astype(np.int32)
        if (i0 // batch) % 2 == 0:
            print(f"[kNN] processed {i1:,}/{n:,}")
    return I_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--public_tsv", required=True)
    ap.add_argument("--sensitive_dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--sensitive_files", required=True,
                    help="Comma list: has_abundance.tsv,has_body_site.tsv,...")

    # ✅ THREAT MODEL: head_prefix is now OPTIONAL.
    # If empty or not provided, ALL nodes from the public graph are used as candidates.
    # This reflects a realistic adversary who only observes G_pub.
    ap.add_argument("--head_prefix", default="",
                    help="Optional prefix filter on candidate heads (e.g. 'Person_'). "
                         "If empty, ALL nodes from the public graph are used (recommended).")

    # struct v3 params
    ap.add_argument("--hash_dim_out", type=int, default=256)
    ap.add_argument("--hash_dim_in", type=int, default=256)
    ap.add_argument("--max_out_neighbors", type=int, default=2000)

    # kNN params
    ap.add_argument("--knn_k", type=int, default=120)
    ap.add_argument("--knn_batch", type=int, default=4000)
    ap.add_argument("--use_faiss_gpu", action="store_true")
    ap.add_argument("--faiss_gpu_id", type=int, default=0)

    # threat model
    ap.add_argument("--seed_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)

    # prediction policy
    ap.add_argument("--max_pred_per_head", type=int, default=1)
    ap.add_argument("--vote_threshold", type=float, default=0.0)
    ap.add_argument("--skip_one_to_one", action="store_true")

    # ranking metrics
    ap.add_argument("--hits_k", type=int, default=10)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    public_tsv = Path(args.public_tsv)
    sens_dir = Path(args.sensitive_dir)
    sens_files = [x.strip() for x in args.sensitive_files.split(",") if x.strip()]
    prefix = args.head_prefix.strip()

    # ----------------------------
    # ✅ THREAT MODEL (CORRECTED)
    # Enumerate candidate heads FROM THE PUBLIC GRAPH ONLY.
    # The adversary has zero knowledge of the sensitive relations at this stage.
    # If head_prefix is provided, filter by prefix (e.g. HealthKG: "Person_").
    # If head_prefix is empty, use ALL nodes (recommended for NELL, Synthea, Freebase, HealthKG).
    # ----------------------------
    print("[ThreatModel] Enumerating candidate heads from public graph only ...")
    heads_union = set()
    for h, r, t in read_public_triples(public_tsv):
        heads_union.add(h)
        heads_union.add(t)

    if prefix:
        before = len(heads_union)
        heads_union = {h for h in heads_union if h.startswith(prefix)}
        print(f"[ThreatModel] Prefix filter '{prefix}': {before:,} -> {len(heads_union):,} heads")
    else:
        print(f"[ThreatModel] No prefix filter: using all {len(heads_union):,} public nodes as candidates")

    # ----------------------------
    # Load sensitive GT (for EVALUATION only — NOT used to define candidate heads)
    # ----------------------------
    gt_Hmap_by_rel = {}
    rels = []

    for fn in sens_files:
        f = sens_dir / fn
        if not f.exists():
            raise SystemExit(f"Missing sensitive file: {f}")

        rel = rel_from_filename(fn)
        H = defaultdict(set)
        tf = Counter()
        lines = 0

        for h, _, t in read_sensitive_flexible(f, rel):
            # GT is loaded for ALL heads (no prefix filter here),
            # evaluation will naturally intersect with heads_union.
            H[h].add(t)
            tf[t] += 1
            lines += 1

        gt_Hmap_by_rel[rel] = H
        rels.append(rel)
        print(f"[GT] {rel}: edges={sum(len(v) for v in H.values())} heads={len(H)} tails={len(tf)} file={fn} lines={lines}")

    if not rels:
        raise SystemExit("No sensitive relations loaded.")

    if args.skip_one_to_one:
        kept = []
        for r in rels:
            _, _, rtype = relation_cardinality_stats(gt_Hmap_by_rel[r])
            if rtype == "1-1":
                print(f"[AutoSkip] {r} Type=1-1")
            else:
                kept.append(r)
        rels = kept
        if not rels:
            raise SystemExit("All relations skipped by --skip_one_to_one.")
    print(f"[Relations] using: {rels}")

    # ----------------------------
    # Build struct v3 for all candidate heads
    # ----------------------------
    struct_csv = outdir / "struct_heads_public_v3.csv"
    df_struct = build_struct_v3(
        public_tsv=public_tsv,
        heads_focus=heads_union,
        out_csv=struct_csv,
        hash_dim_out=args.hash_dim_out,
        hash_dim_in=args.hash_dim_in,
        max_out_neighbors=args.max_out_neighbors
    )

    heads_all = df_struct["head"].astype(str).tolist()
    X_all = df_struct.drop(columns=["head"]).to_numpy(dtype=np.float32)
    head2i = {h: i for i, h in enumerate(heads_all)}

    # heads_universe = candidates that also appear in at least one sensitive GT
    # (used only for the seed/hidden split — the attack itself targets all heads_union)
    gt_heads_all = set()
    for r in rels:
        gt_heads_all |= set(gt_Hmap_by_rel[r].keys())

    heads_universe = [h for h in heads_all if h in gt_heads_all]
    print(f"[Universe] total_candidate_heads={len(heads_all):,} "
          f"heads_with_sensitive_GT={len(heads_universe):,} "
          f"feature_dim={X_all.shape[1]:,}")

    # ----------------------------
    # Split seed/hidden
    # ----------------------------
    H_arr = np.array(heads_universe, dtype=object)
    rng.shuffle(H_arr)
    n_seed = max(1, int(round(args.seed_frac * len(H_arr))))
    seed_heads = set(H_arr[:n_seed].tolist())
    hidden_heads = [h for h in H_arr[n_seed:].tolist()]
    print(f"[Split] seed_heads={len(seed_heads)} hidden_heads={len(hidden_heads)} seed_frac={args.seed_frac}")

    # ----------------------------
    # Prepare normalized vectors
    # ----------------------------
    X_seed = np.vstack([X_all[head2i[h]] for h in seed_heads]).astype(np.float32)
    mu, sd = zfit(X_seed)
    Xs = zapply(X_all, mu, sd).astype(np.float32)
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    Xn = np.ascontiguousarray(Xn, dtype=np.float32)

    # ----------------------------
    # kNN over ALL candidate heads (not just GT heads)
    # ----------------------------
    k = max(1, args.knn_k)
    print(f"[kNN] computing k={k} neighbors over n={Xn.shape[0]:,} "
          f"(faiss={_HAS_FAISS}, use_gpu_flag={args.use_faiss_gpu})")

    if _HAS_FAISS:
        I_all = knn_faiss(Xn, k=k, use_gpu=args.use_faiss_gpu, gpu_id=args.faiss_gpu_id)
    else:
        I_all = knn_batched_cosine(Xn, k=k, batch=args.knn_batch)

    idx2head = heads_all

    # ----------------------------
    # Seed labels per relation
    # ----------------------------
    seed_labels = {}
    for r in rels:
        Hmap = gt_Hmap_by_rel[r]
        lab = {}
        for h in seed_heads:
            if h in Hmap:
                lab[h] = list(Hmap[h])
        seed_labels[r] = lab
        print(f"[Seeds] {r}: labeled_seed_heads={len(lab)}")

    # ----------------------------
    # True edges for hidden heads
    # ----------------------------
    true_edges_by_rel = {}
    for r in rels:
        Hmap = gt_Hmap_by_rel[r]
        T = set()
        for h in hidden_heads:
            for t in Hmap.get(h, set()):
                T.add((h, r, t))
        true_edges_by_rel[r] = T

    # ----------------------------
    # Predict: weighted kNN voting + PR-AUC logging + ranking metrics
    # ----------------------------
    pred_edges_by_rel = {}
    scores_by_rel = {}
    votes_by_rel  = {}

    for r in rels:
        lab = seed_labels[r]
        preds = set()
        scores = []
        votes_by_head = {}

        for h in hidden_heads:
            h_idx = head2i[h]
            neigh_idx = I_all[h_idx]
            w = (Xn[neigh_idx] @ Xn[h_idx]).astype(np.float32)

            votes = defaultdict(float)
            for j, nb_i in enumerate(neigh_idx):
                nb_h = idx2head[int(nb_i)]
                if nb_h not in lab:
                    continue
                for t in lab[nb_h]:
                    votes[t] += float(w[j])

            if not votes:
                continue

            votes_by_head[h] = dict(votes)

            true_tails = gt_Hmap_by_rel[r].get(h, set())
            for t_cand, sc_cand in votes.items():
                label = 1 if t_cand in true_tails else 0
                scores.append((float(sc_cand), int(label)))

            best = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:args.max_pred_per_head]
            if args.vote_threshold > 0.0 and best and best[0][1] < args.vote_threshold:
                continue

            for t, sc in best:
                preds.add((h, r, t))

        pred_edges_by_rel[r] = preds
        scores_by_rel[r] = scores
        votes_by_rel[r] = votes_by_head
        print(f"[Recon] {r}: predicted_edges={len(preds)} heads_with_candidates={len(votes_by_head)}")

    # ----------------------------
    # PR-AUC (candidate-based) per relation + GLOBAL
    # ----------------------------
    pr_auc_by_rel = {}
    all_pairs = []
    for r in rels:
        pairs = scores_by_rel.get(r, [])
        if not pairs:
            pr_auc_by_rel[r] = np.nan
            continue
        y_score = np.array([s for s, _ in pairs], dtype=np.float32)
        y_true  = np.array([l for _, l in pairs], dtype=np.int32)
        pr_auc_by_rel[r] = pr_auc_from_scores(y_true, y_score)
        all_pairs.extend(pairs)

    if all_pairs:
        y_score_g = np.array([s for s, _ in all_pairs], dtype=np.float32)
        y_true_g  = np.array([l for _, l in all_pairs], dtype=np.int32)
        pr_auc_global = pr_auc_from_scores(y_true_g, y_score_g)
    else:
        pr_auc_global = np.nan

    # ----------------------------
    # Ranking metrics per relation + GLOBAL
    # ----------------------------
    K = int(args.hits_k)
    rank_by_rel = {}
    global_votes = {}
    global_true  = {}

    for r in rels:
        true_tails_by_head = {h: gt_Hmap_by_rel[r].get(h, set()) for h in hidden_heads}
        rm = ranking_metrics_from_votes(votes_by_rel.get(r, {}), true_tails_by_head, Ks=(K,))
        rank_by_rel[r] = rm

        for h, votes in votes_by_rel.get(r, {}).items():
            global_votes[(r, h)] = votes
            global_true[(r, h)]  = true_tails_by_head.get(h, set())

    rank_global = ranking_metrics_from_votes(global_votes, global_true, Ks=(K,))

    if K != 10:
        for r in rels:
            if f"Hits@{K}" in rank_by_rel[r]:
                rank_by_rel[r]["Hits@10"] = rank_by_rel[r][f"Hits@{K}"]
        if f"Hits@{K}" in rank_global:
            rank_global["Hits@10"] = rank_global[f"Hits@{K}"]

    # ----------------------------
    # Save outputs
    # ----------------------------
    all_preds = []
    for r in rels:
        all_preds.extend(list(pred_edges_by_rel.get(r, set())))
    df_pred = pd.DataFrame(all_preds, columns=["head","rel","tail"])
    pred_path = outdir / "reconstructed_private_graph.tsv"
    df_pred.to_csv(pred_path, sep="\t", index=False, header=False)

    df_metrics = build_metrics_table(
        true_edges_by_rel, pred_edges_by_rel, rels, gt_Hmap_by_rel,
        pr_auc_by_rel=pr_auc_by_rel, pr_auc_global=pr_auc_global,
        rank_by_rel=rank_by_rel, rank_global=rank_global
    )
    metrics_path = outdir / "reconstruction_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)

    rows_rank = []
    for r in rels:
        row = {"Relation": r}
        row.update(rank_by_rel.get(r, {}))
        rows_rank.append(row)
    row_g = {"Relation": "GLOBAL"}
    row_g.update(rank_global)
    rows_rank.append(row_g)
    df_rank = pd.DataFrame(rows_rank)
    rank_path = outdir / "ranking_metrics.csv"
    df_rank.to_csv(rank_path, index=False)

    meta = {
        "task": "attack3_healthkg_struct_only_knn_voting_full",
        "threat_model": {
            "candidate_heads_source": "public_graph_only",
            "head_prefix_filter": prefix if prefix else "none (all nodes)",
            "note": "Adversary enumerates candidates solely from G_pub. "
                    "No knowledge of sensitive relations used to define target set."
        },
        "public_tsv": str(public_tsv),
        "relations": rels,
        "head_prefix": prefix,
        "seed_frac": args.seed_frac,
        "knn_k": args.knn_k,
        "vote_threshold": args.vote_threshold,
        "max_pred_per_head": args.max_pred_per_head,
        "struct": {
            "hash_dim_out": args.hash_dim_out,
            "hash_dim_in": args.hash_dim_in,
            "max_out_neighbors": args.max_out_neighbors
        },
        "knn_backend": "faiss" if _HAS_FAISS else "batched_cosine",
        "faiss_gpu_used": bool(_HAS_FAISS and args.use_faiss_gpu),
        "faiss_gpu_id": int(args.faiss_gpu_id),
        "pr_auc": {
            "per_relation": pr_auc_by_rel,
            "global": pr_auc_global
        },
        "ranking": {
            "hits_k": K,
            "per_relation": rank_by_rel,
            "global": rank_global
        }
    }
    (outdir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Done.")
    print("  struct_csv        :", struct_csv)
    print("  reconstructed KG  :", pred_path)
    print("  metrics table     :", metrics_path)
    print("  ranking metrics   :", rank_path)
    print("\n=== Reconstruction metrics (hidden heads) ===")
    print(df_metrics.to_string(index=False))
    print("\n=== Ranking metrics (candidate-set) ===")
    print(df_rank.to_string(index=False))


if __name__ == "__main__":
    main()