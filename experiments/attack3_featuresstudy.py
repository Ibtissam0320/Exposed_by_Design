#!/usr/bin/env python3
"""
Attack3 Ablation - ULTRA-FAST with NetworkX + PyTorch GPU
50K heads in 30-60 min (NO FAISS CRASHES!)
"""

import argparse, json, math, random
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import unquote

import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, auc


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


def read_public_triples(path: Path):
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


def pr_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray):
    if y_true.size == 0:
        return np.nan
    pos = int(y_true.sum())
    if pos == 0 or pos == y_true.size:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


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


# ============================================
# ULTRA-FAST FEATURES with NetworkX
# ============================================
def load_graphs_networkx(public_path: Path):
    """Build NetworkX graphs from public triples."""
    print(f"[+] Loading public graph with NetworkX: {public_path}")
    
    G_out = nx.DiGraph()
    G_in = nx.DiGraph()
    G_und = nx.Graph()
    
    n = 0
    for h, r, t in read_public_triples(public_path):
        G_out.add_edge(h, t)
        G_in.add_edge(t, h)
        G_und.add_edge(h, t)
        n += 1
        if n % 1000000 == 0:
            print(f"  Loaded {n:,} triples...")
    
    print(f"[+] Graphs built: {G_und.number_of_nodes():,} nodes, {G_und.number_of_edges():,} edges")
    return G_out, G_in, G_und


def extract_features_networkx(h, G_out, G_in, G_und, components, max_layer):
    """Extract features using NetworkX BFS (10-50x faster!)."""
    h = str(h)
    feats = []
    
    for comp in components:
        if comp == "ni_head":
            try:
                lengths = nx.single_source_shortest_path_length(G_out, h, cutoff=max_layer)
                for layer in range(1, max_layer + 1):
                    count = sum(1 for n, d in lengths.items() if d == layer)
                    feats.append(float(count))
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "ni_tail":
            try:
                lengths = nx.single_source_shortest_path_length(G_in, h, cutoff=max_layer)
                for layer in range(1, max_layer + 1):
                    count = sum(1 for n, d in lengths.items() if d == layer)
                    feats.append(float(count))
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "Ii":
            try:
                lengths = nx.single_source_shortest_path_length(G_und, h, cutoff=max_layer)
                for layer in range(1, max_layer + 1):
                    layer_nodes = [n for n, d in lengths.items() if d == layer]
                    if len(layer_nodes) > 1:
                        subgraph = G_und.subgraph(layer_nodes)
                        feats.append(float(subgraph.number_of_edges()))
                    else:
                        feats.append(0.0)
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "Ei":
            try:
                lengths = nx.single_source_shortest_path_length(G_und, h, cutoff=max_layer + 1)
                for layer in range(1, max_layer + 1):
                    layer_i = set(n for n, d in lengths.items() if d == layer)
                    layer_ip1 = set(n for n, d in lengths.items() if d == layer + 1)
                    
                    if layer_i and layer_ip1:
                        count = 0
                        for u in layer_i:
                            count += len([v for v in G_und.neighbors(u) if v in layer_ip1])
                        feats.append(float(count))
                    else:
                        feats.append(0.0)
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "Ii_head":
            try:
                lengths = nx.single_source_shortest_path_length(G_out, h, cutoff=max_layer)
                for layer in range(1, max_layer + 1):
                    layer_nodes = [n for n, d in lengths.items() if d == layer]
                    if len(layer_nodes) > 1:
                        subgraph = G_out.subgraph(layer_nodes)
                        feats.append(float(subgraph.number_of_edges()))
                    else:
                        feats.append(0.0)
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "Ii_tail":
            try:
                lengths = nx.single_source_shortest_path_length(G_in, h, cutoff=max_layer)
                for layer in range(1, max_layer + 1):
                    layer_nodes = [n for n, d in lengths.items() if d == layer]
                    if len(layer_nodes) > 1:
                        subgraph = G_in.subgraph(layer_nodes)
                        feats.append(float(subgraph.number_of_edges()))
                    else:
                        feats.append(0.0)
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "Ei_head":
            try:
                lengths = nx.single_source_shortest_path_length(G_out, h, cutoff=max_layer + 1)
                for layer in range(1, max_layer + 1):
                    layer_i = set(n for n, d in lengths.items() if d == layer)
                    layer_ip1 = set(n for n, d in lengths.items() if d == layer + 1)
                    
                    if layer_i and layer_ip1:
                        count = 0
                        for u in layer_i:
                            count += len([v for v in G_out.neighbors(u) if v in layer_ip1])
                        feats.append(float(count))
                    else:
                        feats.append(0.0)
            except:
                feats.extend([0.0] * max_layer)
        
        elif comp == "Ei_tail":
            try:
                lengths = nx.single_source_shortest_path_length(G_in, h, cutoff=max_layer + 1)
                for layer in range(1, max_layer + 1):
                    layer_i = set(n for n, d in lengths.items() if d == layer)
                    layer_ip1 = set(n for n, d in lengths.items() if d == layer + 1)
                    
                    if layer_i and layer_ip1:
                        count = 0
                        for u in layer_i:
                            count += len([v for v in G_in.neighbors(u) if v in layer_ip1])
                        feats.append(float(count))
                    else:
                        feats.append(0.0)
            except:
                feats.extend([0.0] * max_layer)
    
    return feats


def build_combined_features(head_ids, G_out, G_in, G_und, feature_combination, max_layer):
    """Build features matrix using NetworkX."""
    print(f"[+] Building features: {feature_combination}, layers 1-{max_layer}")
    print(f"    Processing {len(head_ids):,} heads with NetworkX (FAST!)")
    
    if feature_combination == "all":
        components = ["ni_head", "ni_tail", "Ii", "Ei"]
    else:
        components = [c.strip() for c in feature_combination.split("+")]
    
    print(f"    Components: {components}")
    
    X_rows = []
    for h in tqdm(head_ids, desc="    Extracting"):
        feats = extract_features_networkx(h, G_out, G_in, G_und, components, max_layer)
        X_rows.append(feats)
    
    X = np.array(X_rows, dtype=np.float32)
    print(f"    Feature matrix: {X.shape}")
    return X


# ============================================
# kNN with PyTorch GPU (NO FAISS!)
# ============================================
def zfit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def zapply(X, mu, sd):
    return (X - mu) / sd


def knn_pytorch_gpu(X: np.ndarray, k: int, batch_size: int = 5000, gpu_id: int = 0):
    """
    kNN using PyTorch on GPU (MUCH MORE STABLE than FAISS).
    Processes in batches to avoid OOM.
    """
    device = f"cuda:{gpu_id}"
    print(f"[kNN] PyTorch GPU: {device}, batch={batch_size}, k={k}")
    
    n, d = X.shape
    X_tensor = torch.from_numpy(X).float().to(device)
    
    # Normalize for cosine similarity
    X_norm = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-8)
    
    I_all = np.zeros((n, k), dtype=np.int32)
    
    # Process in batches
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        
        # Compute similarities for this batch
        batch = X_norm[i:end_i]
        sim = torch.mm(batch, X_norm.T)  # (batch_size, n)
        
        # Mask self-similarity
        for j in range(sim.shape[0]):
            sim[j, i + j] = -1e9
        
        # Get top-k
        _, indices = torch.topk(sim, k, dim=1)
        I_all[i:end_i] = indices.cpu().numpy().astype(np.int32)
        
        if (i // batch_size) % 10 == 0:
            print(f"    kNN batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size}")
    
    return I_all


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--public_tsv", required=True)
    ap.add_argument("--sensitive_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sensitive_files", required=True)
    ap.add_argument("--head_prefix", default="")

    ap.add_argument("--feature-combination", type=str, required=True)
    ap.add_argument("--max-layer", type=int, required=True)

    ap.add_argument("--knn_k", type=int, default=30)
    ap.add_argument("--use_faiss_gpu", action="store_true")  # Kept for compatibility
    ap.add_argument("--faiss_gpu_id", type=int, default=0)

    ap.add_argument("--seed_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_pred_per_head", type=int, default=1)
    ap.add_argument("--vote_threshold", type=float, default=0.0)
    ap.add_argument("--hits_k", type=int, default=10)
    ap.add_argument("--max-heads-sample", type=int, default=None)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    outdir = Path(args.outdir).expanduser().resolve()
    
    comb_safe = args.feature_combination.replace("+", "_")
    feature_dir = outdir / comb_safe
    scores_dir = feature_dir / "scores"
    metrics_dir = feature_dir / "metrics"
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"attack3_{comb_safe}_L{args.max_layer}_k{args.knn_k}"
    pred_path = scores_dir / f"{exp_name}_reconstructed.tsv"
    metrics_csv = metrics_dir / f"{exp_name}_metrics.csv"
    rank_csv = metrics_dir / f"{exp_name}_ranking.csv"
    config_json = metrics_dir / f"{exp_name}_config.json"

    public_tsv = Path(args.public_tsv)
    sens_dir = Path(args.sensitive_dir)
    sens_files = [x.strip() for x in args.sensitive_files.split(",") if x.strip()]
    prefix = args.head_prefix.strip()

    # Load graphs with NetworkX
    G_out, G_in, G_und = load_graphs_networkx(public_tsv)

    # Enumerate heads
    print("[ThreatModel] Enumerating candidate heads...")
    heads_union = set(G_und.nodes())
    
    if prefix:
        before = len(heads_union)
        heads_union = {h for h in heads_union if h.startswith(prefix)}
        print(f"[ThreatModel] Prefix '{prefix}': {before:,} -> {len(heads_union):,}")
    else:
        print(f"[ThreatModel] No prefix: {len(heads_union):,} nodes")
    
    if args.max_heads_sample and len(heads_union) > args.max_heads_sample:
        print(f"⚡ SAMPLING {args.max_heads_sample:,} heads (from {len(heads_union):,})")
        heads_union = set(random.sample(list(heads_union), args.max_heads_sample))

    # Load GT
    gt_Hmap_by_rel = {}
    rels = []

    for fn in sens_files:
        f = sens_dir / fn
        if not f.exists():
            raise SystemExit(f"Missing: {f}")

        rel = rel_from_filename(fn)
        H = defaultdict(set)

        for h, _, t in read_sensitive_flexible(f, rel):
            H[h].add(t)

        gt_Hmap_by_rel[rel] = H
        rels.append(rel)
        print(f"[GT] {rel}: edges={sum(len(v) for v in H.values())} heads={len(H)}")

    if not rels:
        raise SystemExit("No relations loaded.")

    # Build features with NetworkX
    heads_all = sorted(heads_union)
    X_all = build_combined_features(
        head_ids=heads_all,
        G_out=G_out,
        G_in=G_in,
        G_und=G_und,
        feature_combination=args.feature_combination,
        max_layer=args.max_layer,
    )

    head2i = {h: i for i, h in enumerate(heads_all)}

    gt_heads_all = set()
    for r in rels:
        gt_heads_all |= set(gt_Hmap_by_rel[r].keys())

    heads_universe = [h for h in heads_all if h in gt_heads_all]
    print(f"[Universe] candidates={len(heads_all):,} with_GT={len(heads_universe):,}")

    # Split
    H_arr = np.array(heads_universe, dtype=object)
    rng.shuffle(H_arr)
    n_seed = max(1, int(round(args.seed_frac * len(H_arr))))
    seed_heads = set(H_arr[:n_seed].tolist())
    hidden_heads = [h for h in H_arr[n_seed:].tolist()]
    print(f"[Split] seed={len(seed_heads)} hidden={len(hidden_heads)}")

    # Normalize
    X_seed = np.vstack([X_all[head2i[h]] for h in seed_heads]).astype(np.float32)
    mu, sd = zfit(X_seed)
    Xs = zapply(X_all, mu, sd).astype(np.float32)
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    Xn = np.ascontiguousarray(Xn, dtype=np.float32)

    # kNN with PyTorch GPU (NO FAISS!)
    k = max(1, args.knn_k)
    print(f"[kNN] k={k}, n={Xn.shape[0]:,}")
    
    I_all = knn_pytorch_gpu(Xn, k=k, batch_size=5000, gpu_id=args.faiss_gpu_id)

    # Seed labels
    seed_labels = {}
    for r in rels:
        Hmap = gt_Hmap_by_rel[r]
        lab = {}
        for h in seed_heads:
            if h in Hmap:
                lab[h] = list(Hmap[h])
        seed_labels[r] = lab
        print(f"[Seeds] {r}: {len(lab)}")

    # True edges
    true_edges_by_rel = {}
    for r in rels:
        Hmap = gt_Hmap_by_rel[r]
        T = set()
        for h in hidden_heads:
            for t in Hmap.get(h, set()):
                T.add((h, r, t))
        true_edges_by_rel[r] = T

    # Predict
    pred_edges_by_rel = {}
    scores_by_rel = {}
    votes_by_rel = {}

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
                nb_h = heads_all[int(nb_i)]
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
        print(f"[Recon] {r}: {len(preds)} edges")

    # PR-AUC
    pr_auc_by_rel = {}
    all_pairs = []
    for r in rels:
        pairs = scores_by_rel.get(r, [])
        if not pairs:
            pr_auc_by_rel[r] = np.nan
            continue
        y_score = np.array([s for s, _ in pairs], dtype=np.float32)
        y_true = np.array([l for _, l in pairs], dtype=np.int32)
        pr_auc_by_rel[r] = pr_auc_from_scores(y_true, y_score)
        all_pairs.extend(pairs)

    if all_pairs:
        y_score_g = np.array([s for s, _ in all_pairs], dtype=np.float32)
        y_true_g = np.array([l for _, l in all_pairs], dtype=np.int32)
        pr_auc_global = pr_auc_from_scores(y_true_g, y_score_g)
    else:
        pr_auc_global = np.nan

    # Ranking
    K = int(args.hits_k)
    rank_by_rel = {}
    global_votes = {}
    global_true = {}

    for r in rels:
        true_tails_by_head = {h: gt_Hmap_by_rel[r].get(h, set()) for h in hidden_heads}
        rm = ranking_metrics_from_votes(votes_by_rel.get(r, {}), true_tails_by_head, Ks=(K,))
        rank_by_rel[r] = rm

        for h, votes in votes_by_rel.get(r, {}).items():
            global_votes[(r, h)] = votes
            global_true[(r, h)] = true_tails_by_head.get(h, set())

    rank_global = ranking_metrics_from_votes(global_votes, global_true, Ks=(K,))

    # Save
    all_preds = []
    for r in rels:
        all_preds.extend(list(pred_edges_by_rel.get(r, set())))
    df_pred = pd.DataFrame(all_preds, columns=["head", "rel", "tail"])
    df_pred.to_csv(pred_path, sep="\t", index=False, header=False)

    df_metrics = build_metrics_table(
        true_edges_by_rel, pred_edges_by_rel, rels, gt_Hmap_by_rel,
        pr_auc_by_rel=pr_auc_by_rel, pr_auc_global=pr_auc_global,
        rank_by_rel=rank_by_rel, rank_global=rank_global
    )
    df_metrics.to_csv(metrics_csv, index=False)

    rows_rank = []
    for r in rels:
        row = {"Relation": r}
        row.update(rank_by_rel.get(r, {}))
        rows_rank.append(row)
    df_rank = pd.DataFrame(rows_rank)
    df_rank.to_csv(rank_csv, index=False)

    meta = {
        "attack": "attack3_networkx_pytorch",
        "feature_combination": args.feature_combination,
        "max_layer": args.max_layer,
        "knn_k": args.knn_k,
        "num_features": int(X_all.shape[1]),
        "seed_frac": args.seed_frac,
        "pr_auc_global": pr_auc_global,
        "mrr_global": rank_global.get("MRR", np.nan),
        "hits@10_global": rank_global.get(f"Hits@{K}", np.nan),
    }

    with open(config_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Done: {exp_name}")
    print(f"  PR-AUC: {pr_auc_global:.4f}")
    print(f"  Files: {metrics_csv}")


if __name__ == "__main__":
    main()