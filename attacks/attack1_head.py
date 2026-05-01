import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_public_graph(public_path: Path, has_header: bool):
    """
    Load graph building TWO structures:
      - out_neighbors: directional h -> {t}   (used for ni_head)
      - neighbors_with_rel: undirected node -> [(neighbor, relation)]  (used for Ri)
    """
    print(f"[+] Loading public graph from {public_path}")

    if has_header:
        df = pd.read_csv(public_path, sep="\t", header=0,
                         names=["head_id", "rel_id", "tail_id"], dtype=str).dropna()
    else:
        df = pd.read_csv(public_path, sep="\t", header=None,
                         names=["head_id", "rel_id", "tail_id"], dtype=str).dropna()

    print(f"    Loaded {len(df):,} triples")

    out_neighbors = defaultdict(set)          # directional: h -> {t}
    neighbors_with_rel = defaultdict(list)    # undirected:  node -> [(neighbor, rel)]

    for row in df.itertuples(index=False):
        h, r, t = str(row.head_id), str(row.rel_id), str(row.tail_id)
        # Directional (for ni_head)
        out_neighbors[h].add(t)
        # Undirected (for Ri — same as attack1)
        neighbors_with_rel[h].append((t, r))
        neighbors_with_rel[t].append((h, r))

    print(f"    Directional graph : {len(out_neighbors):,} nodes")
    print(f"    Undirected graph  : {len(neighbors_with_rel):,} nodes")
    return out_neighbors, neighbors_with_rel


def load_sensitive_heads(sens_path: Path):
    print(f"[+] Loading sensitive heads from {sens_path}")
    df = pd.read_csv(sens_path, sep="\t", header=None,
                     names=["head_id", "rel_id", "tail_id"], dtype=str).dropna()
    pos_heads = sorted(set(df["head_id"].astype(str).unique().tolist()))
    print(f"    Positive heads: {len(pos_heads):,}")
    return pos_heads


# ─────────────────────────────────────────────────────────────────
# 2. FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────

def bfs_directional_layers(start, out_neighbors, max_hop):
    """
    Directional BFS (OUT edges only).
    Returns dict: layer_index -> set of nodes in that layer.
    Used for ni_head.
    """
    start = str(start)
    layers = {}
    seen = {start}
    frontier = {start}

    for i in range(1, max_hop + 1):
        next_layer = set()
        for u in frontier:
            for v in out_neighbors.get(u, set()):
                v = str(v)
                if v not in seen:
                    seen.add(v)
                    next_layer.add(v)
        layers[i] = next_layer
        frontier = next_layer
        if not frontier:
            break

    return layers


def bfs_undirected_ri_per_layer(start, neighbors_with_rel, max_hop):
    """
    Undirected BFS that counts UNIQUE relation types discovered at each layer.
    Exactly the same logic as attack1_Ri_only.py.

    Returns: list of Ri values [Ri_L1, Ri_L2, ..., Ri_Lmax]
    """
    start = str(start)
    seen = {start}
    frontier = {start}
    Ri_values = []

    for hop in range(1, max_hop + 1):
        next_layer = set()
        relations_in_layer = set()  # unique relation types discovered this layer

        for u in frontier:
            for (v, rel) in neighbors_with_rel.get(u, []):
                v = str(v)
                if v not in seen:
                    seen.add(v)
                    next_layer.add(v)
                    relations_in_layer.add(rel)

        Ri_values.append(float(len(relations_in_layer)))
        frontier = next_layer

        if not frontier:
            Ri_values.extend([0.0] * (max_hop - hop))
            break

    return Ri_values


def extract_features(h, out_neighbors, neighbors_with_rel, max_hop):
    """
    Extract [ni_head_L1, Ri_L1, ni_head_L2, Ri_L2, ...] for node h.

      ni_head_Li : size of layer i in directional BFS
      Ri_Li      : unique relation types discovered at layer i in undirected BFS
    """
    h = str(h)

    # Directional BFS for ni_head
    dir_layers = bfs_directional_layers(h, out_neighbors, max_hop)

    # Undirected BFS for Ri (same as attack1)
    ri_values = bfs_undirected_ri_per_layer(h, neighbors_with_rel, max_hop)

    feats = []
    for i in range(1, max_hop + 1):
        ni = float(len(dir_layers.get(i, set())))
        ri = ri_values[i - 1] if i - 1 < len(ri_values) else 0.0
        feats.append(ni)
        feats.append(ri)

    return feats


def build_features(head_ids, out_neighbors, neighbors_with_rel, max_hop):
    print(f"[+] Extracting ni_head + Ri features for {len(head_ids):,} heads")
    print(f"    Max hops : {max_hop}")
    print(f"    Features : {max_hop * 2}  (ni_head + Ri per layer)")

    X_rows = []
    for h in head_ids:
        feats = extract_features(h, out_neighbors, neighbors_with_rel, max_hop)
        X_rows.append(feats)

    X = np.array(X_rows, dtype=float)
    print(f"    Feature matrix: {X.shape}")

    print(f"\n    Per-feature statistics:")
    feat_names = []
    for i in range(1, max_hop + 1):
        feat_names += [f"ni_head_L{i}", f"Ri_L{i}"]
    for j, name in enumerate(feat_names):
        col = X[:, j]
        print(f"      {name:15s}: mean={col.mean():.2f}, std={col.std():.2f}, "
              f"min={col.min():.0f}, max={col.max():.0f}, "
              f"non-zero={100*(col>0).sum()/len(col):.1f}%")

    return X


# ─────────────────────────────────────────────────────────────────
# 3. NEGATIVE SAMPLING
# ─────────────────────────────────────────────────────────────────

def compute_out_degrees(out_neighbors):
    return {node: len(neighs) for node, neighs in out_neighbors.items()}


def select_hard_negatives(all_heads, pos_set, exclude_set,
                          out_degrees, num_samples, mode, rng):
    """Select hard negatives based on OUT degree."""
    candidates = [h for h in all_heads
                  if h not in pos_set and h not in exclude_set]

    if not candidates:
        return np.array([], dtype=object)

    degs = [out_degrees.get(h, 0) for h in candidates]
    median_deg = np.median(degs)
    print(f"    Hard negatives: median OUT degree = {median_deg:.1f}")

    if mode == "median_ge":
        filtered = [h for h, d in zip(candidates, degs) if d >= median_deg]
    elif mode == "median_band":
        lo, hi = median_deg * 0.75, median_deg * 1.25
        filtered = [h for h, d in zip(candidates, degs) if lo <= d <= hi]
    else:
        filtered = candidates

    if len(filtered) < num_samples:
        print(f"    Warning: only {len(filtered)} hard candidates, using all negatives")
        filtered = candidates

    return rng.choice(filtered, size=min(num_samples, len(filtered)), replace=False)


# ─────────────────────────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test,
                device, batch_size=512, epochs=100, lr=0.001):
    print(f"[+] Training MLP on {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t  = torch.FloatTensor(X_test).to(device)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                        batch_size=batch_size, shuffle=True)

    model     = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        y_scores = model(X_test_t).cpu().numpy().flatten()

    return y_scores


# ─────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens-path",         type=str,   required=True)
    ap.add_argument("--public-path",       type=str,   required=True)
    ap.add_argument("--public-has-header", action="store_true")
    ap.add_argument("--max-hop",           type=int,   default=2)
    ap.add_argument("--num-neg",           type=int,   default=5000)
    ap.add_argument("--hard-neg-mode",     type=str,   default="median_ge",
                    choices=["median_ge", "median_band"])
    ap.add_argument("--outdir",            type=str,   required=True)
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--device",            type=str,   default="cuda")
    ap.add_argument("--batch-size",        type=int,   default=512)
    ap.add_argument("--epochs",            type=int,   default=100)
    ap.add_argument("--lr",                type=float, default=0.001)
    args = ap.parse_args()

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[+] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[+] Using CPU")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    public_path = Path(args.public_path).expanduser()
    sens_path   = Path(args.sens_path).expanduser()
    outdir      = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    out_neighbors, neighbors_with_rel = load_public_graph(
        public_path, args.public_has_header)
    pos_heads_all = load_sensitive_heads(sens_path)

    all_heads = np.array(sorted(out_neighbors.keys()), dtype=object)
    pos_set   = set(pos_heads_all)

    # Train/test split
    num_pos = len(pos_heads_all)
    perm    = rng.permutation(num_pos)
    n_train = max(1, int(0.2 * num_pos))
    train_pos = np.array(pos_heads_all)[perm[:n_train]]
    test_pos  = np.array(pos_heads_all)[perm[n_train:]]

    # Negatives (no overlap)
    out_degrees = compute_out_degrees(out_neighbors)
    print(f"\n[+] Selecting hard negatives (no train/test overlap)")

    all_hard_negs = select_hard_negatives(
        all_heads, pos_set, set(train_pos),
        out_degrees, args.num_neg * 2, args.hard_neg_mode, rng)

    if len(all_hard_negs) >= args.num_neg * 2:
        train_neg = all_hard_negs[:args.num_neg]
        test_neg  = all_hard_negs[args.num_neg:args.num_neg * 2]
    else:
        split     = len(all_hard_negs) // 2
        train_neg = all_hard_negs[:split]
        test_neg  = all_hard_negs[split:]

    assert len(set(train_neg) & set(test_neg)) == 0, "Train/Test neg overlap!"

    heads_train = np.concatenate([train_pos, train_neg])
    y_train     = np.concatenate([np.ones(len(train_pos)),  np.zeros(len(train_neg))])
    heads_test  = np.concatenate([test_pos,  test_neg])
    y_test      = np.concatenate([np.ones(len(test_pos)),   np.zeros(len(test_neg))])

    print(f"[+] Train : {len(train_pos)} pos, {len(train_neg)} neg")
    print(f"[+] Test  : {len(test_pos)} pos, {len(test_neg)} neg")

    # Features
    X_train = build_features(heads_train, out_neighbors, neighbors_with_rel, args.max_hop)
    X_test  = build_features(heads_test,  out_neighbors, neighbors_with_rel, args.max_hop)

    # Scale
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # Train
    y_scores = train_model(X_train_sc, y_train, X_test_sc, y_test,
                           device=device, batch_size=args.batch_size,
                           epochs=args.epochs, lr=args.lr)

    # Evaluate
    pr_auc  = float(average_precision_score(y_test, y_scores))
    try:
        roc_auc = float(roc_auc_score(y_test, y_scores))
    except Exception:
        roc_auc = None

    print(f"\n✅ PR-AUC  = {pr_auc:.4f}")
    if roc_auc:
        print(f"✅ ROC-AUC = {roc_auc:.4f}")

    # Save
    exp_name = (f"{sens_path.stem}_ni_head_Ri_undirected"
                f"_L{args.max_hop}_{args.hard_neg_mode}")

    pd.DataFrame({"head_id": heads_test, "label": y_test, "score": y_scores})\
      .to_csv(outdir / f"{exp_name}_scores.tsv", sep="\t", index=False)

    metrics = {
        "relation":            sens_path.stem,
        "feature_group":       "ni_head_Ri_undirected",
        "Ri_computation":      "undirected BFS — unique relation types per layer (same as attack1_Ri_only)",
        "max_hop":             args.max_hop,
        "num_features":        X_train.shape[1],
        "features_per_layer":  2,
        "hard_neg_criterion":  "OUT_degree",
        "device":              str(device),
        "batch_size":          args.batch_size,
        "epochs":              args.epochs,
        "lr":                  args.lr,
        "num_train_pos":       len(train_pos),
        "num_test_pos":        len(test_pos),
        "num_train_neg":       len(train_neg),
        "num_test_neg":        len(test_neg),
        "pr_auc_test":         pr_auc,
        "roc_auc_test":        roc_auc,
    }

    with open(outdir / f"{exp_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Done | ni_head + Ri (undirected) L{args.max_hop} | {device}")


if __name__ == "__main__":
    main()