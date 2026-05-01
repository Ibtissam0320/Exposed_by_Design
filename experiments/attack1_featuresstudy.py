#!/usr/bin/env python3
"""
ABLATION STUDY - Attack1
Features available:
  - ni_head : OUT neighbors count per layer  (directional, out_neighbors)
  - ni_tail : IN  neighbors count per layer  (directional, in_neighbors)
  - Ii_head : outgoing edges within layer    (directional, out_neighbors, NO div by 2)
  - Ei_head : outgoing edges between layers  (directional, out_neighbors)
  - Ri      : unique relation types per layer (undirected BFS, neighbors_with_rel)

Usage:
  python ablation_attack1.py \
    --sens-path   /path/to/sensitive.tsv \
    --public-path /path/to/public.tsv \
    --feature-group ni_head   --max-layer 2 \
    --outdir ./results_ablation
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ─────────────────────────────────────────────────────────────────
# 1. MODEL
# ─────────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_mlp_gpu(X_train, y_train, X_val, y_val,
                  epochs=100, batch_size=2048, lr=0.001, device='cuda'):
    input_dim = X_train.shape[1]
    model     = MLPClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train).to(device),
                      torch.FloatTensor(y_train).unsqueeze(1).to(device)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val).to(device),
                      torch.FloatTensor(y_val).unsqueeze(1).to(device)),
        batch_size=batch_size, shuffle=False)

    best_val_loss, patience_counter, patience = float('inf'), 0, 10

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        val_loss = sum(criterion(model(xb), yb).item()
                       for xb, yb in val_loader) / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    return model


def predict_mlp_gpu(model, X_test, batch_size=2048, device='cuda'):
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(device)),
                        batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds).flatten()


# ─────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_public_graph(public_path: Path, has_header: bool):
    """
    Build four graph structures:
      - out_neighbors      : h -> {t}               (directional, for ni_head / Ii_head / Ei_head)
      - in_neighbors       : t -> {h}               (directional, for ni_tail)
      - neighbors_und      : node -> {neighbors}    (undirected,  for hard-neg degrees)
      - neighbors_with_rel : node -> [(nbr, rel)]   (undirected,  for Ri)
    """
    print(f"[+] Loading public graph from {public_path}")

    if has_header:
        df = pd.read_csv(public_path, sep="\t", header=0,
                         names=["head_id", "rel_id", "tail_id"],
                         dtype=str, low_memory=False).dropna()
    else:
        df = pd.read_csv(public_path, sep="\t", header=None,
                         names=["head_id", "rel_id", "tail_id"],
                         dtype=str, low_memory=False).dropna()

    print(f"    Loaded {len(df):,} triples")

    out_neighbors      = defaultdict(set)
    in_neighbors       = defaultdict(set)
    neighbors_und      = defaultdict(set)
    neighbors_with_rel = defaultdict(list)

    for row in df.itertuples(index=False):
        h, r, t = str(row.head_id), str(row.rel_id), str(row.tail_id)
        out_neighbors[h].add(t)
        in_neighbors[t].add(h)
        neighbors_und[h].add(t)
        neighbors_und[t].add(h)
        neighbors_with_rel[h].append((t, r))
        neighbors_with_rel[t].append((h, r))

    print(f"    Graph: {len(neighbors_und):,} nodes")
    return out_neighbors, in_neighbors, neighbors_und, neighbors_with_rel


def load_sensitive_heads(sens_path: Path):
    print(f"[+] Loading sensitive heads from {sens_path}")
    df = pd.read_csv(sens_path, sep="\t", header=None,
                     names=["head_id", "rel_id", "tail_id"],
                     dtype=str, low_memory=False).dropna()
    pos_heads = sorted(set(df["head_id"].astype(str).unique().tolist()))
    print(f"    Positive heads: {len(pos_heads):,}")
    return pos_heads


# ─────────────────────────────────────────────────────────────────
# 3. BFS HELPERS
# ─────────────────────────────────────────────────────────────────

def bfs_layers(start, neighbors_dict, max_hop):
    """Generic BFS returning dict layer_index -> set of nodes."""
    start   = str(start)
    seen    = {start}
    frontier = {start}
    layers  = {}
    for i in range(1, max_hop + 1):
        nxt = set()
        for u in frontier:
            for v in neighbors_dict.get(u, set()):
                v = str(v)
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        layers[i] = nxt
        frontier  = nxt
        if not frontier:
            break
    return layers


def bfs_ri(start, neighbors_with_rel, max_hop):
    """
    Undirected BFS counting UNIQUE relation types discovered at each layer.
    Identical to attack1_Ri_only.py logic.
    Returns list [Ri_L1, ..., Ri_Lmax].
    """
    start    = str(start)
    seen     = {start}
    frontier = {start}
    ri_vals  = []

    for hop in range(1, max_hop + 1):
        nxt  = set()
        rels = set()
        for u in frontier:
            for (v, rel) in neighbors_with_rel.get(u, []):
                v = str(v)
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
                    rels.add(rel)
        ri_vals.append(float(len(rels)))
        frontier = nxt
        if not frontier:
            ri_vals.extend([0.0] * (max_hop - hop))
            break

    return ri_vals


# ─────────────────────────────────────────────────────────────────
# 4. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────

def extract_features(node, feature_group, max_hop,
                     out_neighbors, in_neighbors,
                     neighbors_with_rel):
    """
    Extract feature vector for one node.

    ni_head  : [|L1|, |L2|, ...]  via out_neighbors  (directional OUT)
    ni_tail  : [|L1|, |L2|, ...]  via in_neighbors   (directional IN)
    Ii_head  : outgoing intra-layer edges per layer   (directional OUT, NO div by 2)
    Ei_head  : outgoing inter-layer edges per layer   (directional OUT)
    Ri       : unique relation types per layer        (undirected BFS)
    """
    node = str(node)

    if feature_group == 'ni_head':
        layers = bfs_layers(node, out_neighbors, max_hop)
        return [float(len(layers.get(i, set()))) for i in range(1, max_hop + 1)]

    elif feature_group == 'ni_tail':
        layers = bfs_layers(node, in_neighbors, max_hop)
        return [float(len(layers.get(i, set()))) for i in range(1, max_hop + 1)]

    elif feature_group == 'Ii_head':
        # Directional: count outgoing edges whose target is also in the same layer
        # NO division by 2 (directed edges are not double-counted)
        layers = bfs_layers(node, out_neighbors, max_hop)
        feats  = []
        for i in range(1, max_hop + 1):
            layer_set = layers.get(i, set())
            count     = sum(len(out_neighbors.get(u, set()) & layer_set)
                            for u in layer_set)
            feats.append(float(count))
        return feats

    elif feature_group == 'Ei_head':
        # Directional: count outgoing edges from layer i to layer i+1
        layers = bfs_layers(node, out_neighbors, max_hop)
        feats  = []
        for i in range(1, max_hop + 1):
            li   = layers.get(i,     set())
            li1  = layers.get(i + 1, set())
            count = sum(len(out_neighbors.get(u, set()) & li1) for u in li)
            feats.append(float(count))
        return feats

    elif feature_group == 'Ri':
        return bfs_ri(node, neighbors_with_rel, max_hop)

    else:
        raise ValueError(f"Unknown feature_group: {feature_group}")


def build_features(node_ids, feature_group, max_hop,
                   out_neighbors, in_neighbors, neighbors_with_rel):
    print(f"[+] Extracting {feature_group} features for {len(node_ids):,} nodes  "
          f"(max_hop={max_hop})")

    X = np.array(
        [extract_features(n, feature_group, max_hop,
                          out_neighbors, in_neighbors, neighbors_with_rel)
         for n in node_ids],
        dtype=float)

    print(f"    Feature matrix: {X.shape}")
    feat_names = [f"{feature_group}_L{i}" for i in range(1, max_hop + 1)]
    for j, name in enumerate(feat_names):
        col = X[:, j]
        print(f"      {name:15s}: mean={col.mean():.2f}  std={col.std():.2f}  "
              f"min={col.min():.0f}  max={col.max():.0f}  "
              f"non-zero={100*(col>0).sum()/len(col):.1f}%")
    return X


# ─────────────────────────────────────────────────────────────────
# 5. NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────────────

def hardneg_select(mode, neg_candidates, degrees, ref_pos,
                   num_sample, rng, band_alpha=10.0):
    neg_candidates = list(neg_candidates)
    if mode == "bruteforce_all":
        return np.array(neg_candidates, dtype=object)

    pos_degs   = np.array([degrees.get(str(h), 0.0) for h in ref_pos], dtype=float)
    median_deg = float(np.median(pos_degs))

    if mode == "median_ge":
        pool = [h for h in neg_candidates if degrees.get(str(h), 0.0) >= median_deg]
    elif mode == "median_band":
        lo = median_deg - band_alpha
        hi = median_deg + band_alpha
        pool = [h for h in neg_candidates
                if lo <= degrees.get(str(h), 0.0) <= hi]
    else:  # none
        pool = neg_candidates

    if not pool:
        pool = neg_candidates

    print(f"    Hard-neg: mode={mode}  median={median_deg:.1f}  "
          f"pool={len(pool):,}  sample={min(num_sample, len(pool)):,}")
    return rng.choice(np.array(pool, dtype=object),
                      size=min(num_sample, len(pool)), replace=False)


# ─────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────

def safe_filename(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def main():
    ap = argparse.ArgumentParser(description="Attack1 Ablation — single feature group")
    ap.add_argument("--sens-path",          type=str,   required=True)
    ap.add_argument("--public-path",        type=str,   required=True)
    ap.add_argument("--public-has-header",  action="store_true")
    ap.add_argument("--feature-group",      type=str,   required=True,
                    choices=["ni_head", "ni_tail", "Ii_head", "Ei_head", "Ri"])
    ap.add_argument("--max-layer",          type=int,   required=True)
    ap.add_argument("--pos-train-fraction", type=float, default=0.2)
    ap.add_argument("--train-neg-sample",   type=int,   default=5000)
    ap.add_argument("--test-neg-sample",    type=int,   default=5000)
    ap.add_argument("--hardneg-mode",       type=str,   default="median_ge",
                    choices=["median_ge", "median_band", "none", "bruteforce_all"])
    ap.add_argument("--hardneg-band-alpha", type=float, default=10.0)
    ap.add_argument("--outdir",             type=str,   default="./results_ablation")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--device",             type=str,   default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--batch-size",         type=int,   default=2048)
    ap.add_argument("--epochs",             type=int,   default=100)
    args = ap.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = args.device
    print(f"[+] Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    public_path = Path(args.public_path).expanduser()
    sens_path   = Path(args.sens_path).expanduser()
    outdir      = Path(args.outdir).expanduser()

    feat_dir    = outdir / args.feature_group
    metrics_dir = feat_dir / "metrics"
    scores_dir  = feat_dir / "scores"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    exp_name     = f"{safe_filename(sens_path.stem)}_{args.feature_group}_L{args.max_layer}"
    metrics_path = metrics_dir / f"{exp_name}_metrics.json"
    scores_path  = scores_dir  / f"{exp_name}_scores.tsv"

    # ── Load graph ───────────────────────────────────────────────
    out_neighbors, in_neighbors, neighbors_und, neighbors_with_rel = \
        load_public_graph(public_path, args.public_has_header)

    deg_total = {n: len(nb) for n, nb in neighbors_und.items()}

    # ── Load positives ───────────────────────────────────────────
    pos_heads_all = np.array(load_sensitive_heads(sens_path), dtype=object)
    pos_set       = set(map(str, pos_heads_all))
    all_heads     = np.array(sorted(neighbors_und.keys()), dtype=object)
    true_negs_all = np.array([h for h in all_heads if str(h) not in pos_set], dtype=object)

    num_pos = len(pos_heads_all)
    print(f"[+] Positives={num_pos:,}  Negatives available={len(true_negs_all):,}")

    if num_pos < 2:
        raise RuntimeError("Not enough positives to split train/test")

    # ── Train / test split ───────────────────────────────────────
    perm    = rng.permutation(num_pos)
    n_train = max(1, min(int(args.pos_train_fraction * num_pos), num_pos - 1))
    train_pos = pos_heads_all[perm[:n_train]]
    test_pos  = pos_heads_all[perm[n_train:]]

    train_pos_set         = set(map(str, train_pos))
    train_unlabeled_pool  = np.array([h for h in all_heads
                                      if str(h) not in train_pos_set], dtype=object)

    # ── Hard negatives ───────────────────────────────────────────
    print(f"\n[+] Selecting hard negatives")
    train_neg = hardneg_select("median_ge" if args.hardneg_mode == "bruteforce_all"
                               else args.hardneg_mode,
                               train_unlabeled_pool, deg_total, train_pos,
                               args.train_neg_sample, rng, args.hardneg_band_alpha)

    rng_test  = np.random.default_rng(args.seed + 1)
    test_neg  = hardneg_select(args.hardneg_mode,
                               true_negs_all, deg_total, train_pos,
                               args.test_neg_sample, rng_test, args.hardneg_band_alpha)

    # ── Assemble datasets ────────────────────────────────────────
    heads_train = np.concatenate([train_pos, train_neg])
    y_train_full = np.concatenate([np.ones(len(train_pos)),
                                   np.zeros(len(train_neg))]).astype(float)
    heads_test  = np.concatenate([test_pos, test_neg])
    y_test      = np.concatenate([np.ones(len(test_pos)),
                                  np.zeros(len(test_neg))]).astype(float)

    print(f"[+] Train: {len(train_pos)} pos + {len(train_neg)} neg = {len(heads_train)}")
    print(f"[+] Test : {len(test_pos)} pos + {len(test_neg)} neg = {len(heads_test)}")

    # ── Extract features ─────────────────────────────────────────
    X_train_full = build_features(heads_train, args.feature_group, args.max_layer,
                                  out_neighbors, in_neighbors, neighbors_with_rel)
    X_test       = build_features(heads_test,  args.feature_group, args.max_layer,
                                  out_neighbors, in_neighbors, neighbors_with_rel)

    # ── Scale ────────────────────────────────────────────────────
    scaler        = StandardScaler()
    X_train_sc    = scaler.fit_transform(X_train_full)
    X_test_sc     = scaler.transform(X_test)

    # Train / val split
    n_val        = max(1, int(0.1 * len(X_train_sc)))
    perm_train   = rng.permutation(len(X_train_sc))
    X_val_sc     = X_train_sc[perm_train[:n_val]]
    y_val        = y_train_full[perm_train[:n_val]]
    X_train_sc   = X_train_sc[perm_train[n_val:]]
    y_train      = y_train_full[perm_train[n_val:]]

    # ── Train ────────────────────────────────────────────────────
    print(f"\n[+] Training MLP — feature={args.feature_group}  "
          f"layers=1-{args.max_layer}  device={device}")
    model = train_mlp_gpu(X_train_sc, y_train, X_val_sc, y_val,
                          epochs=args.epochs, batch_size=args.batch_size,
                          device=device)

    # ── Evaluate ─────────────────────────────────────────────────
    y_scores = predict_mlp_gpu(model, X_test_sc,
                               batch_size=args.batch_size, device=device)

    pr_auc  = float(average_precision_score(y_test, y_scores))
    try:
        roc_auc = float(roc_auc_score(y_test, y_scores))
    except Exception:
        roc_auc = None

    print(f"\n✅ PR-AUC  = {pr_auc:.4f}")
    if roc_auc:
        print(f"✅ ROC-AUC = {roc_auc:.4f}")

    # ── Save ─────────────────────────────────────────────────────
    pd.DataFrame({"head_id": heads_test, "label": y_test, "score": y_scores}) \
      .to_csv(scores_path, sep="\t", index=False)

    metrics = {
        "relation":          sens_path.stem,
        "feature_group":     args.feature_group,
        "max_layer":         args.max_layer,
        "num_features":      int(X_train_full.shape[1]),
        "num_train_pos":     int(len(train_pos)),
        "num_test_pos":      int(len(test_pos)),
        "num_train_neg":     int(len(train_neg)),
        "num_test_neg":      int(len(test_neg)),
        "pr_auc_test":       pr_auc,
        "roc_auc_test":      roc_auc,
        "hardneg_mode":      args.hardneg_mode,
        "device":            device,
        "note": {
            "ni_head":  "directional OUT BFS, layer sizes",
            "ni_tail":  "directional IN  BFS, layer sizes",
            "Ii_head":  "directional OUT, intra-layer edges, NO div by 2",
            "Ei_head":  "directional OUT, inter-layer edges",
            "Ri":       "undirected BFS, unique relation types per layer",
        }[args.feature_group],
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[+] Scores  → {scores_path}")
    print(f"[+] Metrics → {metrics_path}")
    print("✅ Done.")


if __name__ == "__main__":
    main()