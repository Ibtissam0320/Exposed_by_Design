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
from sklearn.model_selection import train_test_split


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


def load_attack1_scores(scores_path: Path, id_col: str, score_col: str):
    df = pd.read_csv(scores_path, sep="\t", dtype=str).dropna()
    if id_col not in df.columns or score_col not in df.columns:
        raise RuntimeError(f"Attack1 file must have '{id_col}' and '{score_col}'")
    df = df[[id_col, score_col]].dropna()
    df[id_col] = df[id_col].astype(str)
    df[score_col] = df[score_col].astype(float)
    return dict(zip(df[id_col].tolist(), df[score_col].tolist()))


def load_public_graph(public_path: Path, has_header: bool):
    print(f"[+] Loading public graph from {public_path}")
    if has_header:
        df_pub = pd.read_csv(public_path, sep="\t", header=0,
                            names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    else:
        df_pub = pd.read_csv(public_path, sep="\t", header=None,
                            names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    
    print(f"    Loaded {len(df_pub):,} triples")
    neighbors = defaultdict(set)
    for row in df_pub.itertuples(index=False):
        h, t = str(row.head_id), str(row.tail_id)
        neighbors[h].add(t)
        neighbors[t].add(h)
    
    degrees = {node: len(neighs) for node, neighs in neighbors.items()}
    print(f"    Graph: {len(neighbors):,} nodes (UNDIRECTED)")
    return neighbors, degrees


def load_sensitive_triples(sens_path: Path):
    print(f"[+] Loading sensitive triples from {sens_path}")
    df_sens = pd.read_csv(sens_path, sep="\t", header=None,
                         names=["head_id","rel_id","tail_id"], dtype=str).dropna()
    print(f"    Loaded {len(df_sens):,} sensitive triples")
    return df_sens


def bfs_layers_undirected(start, neighbors, max_hop=3):
    start = str(start)
    layers = {}
    seen = {start}
    frontier = {start}
    
    for i in range(1, max_hop + 1):
        next_layer = set()
        for u in frontier:
            for v in neighbors.get(str(u), set()):
                v = str(v)
                if v not in seen:
                    seen.add(v)
                    next_layer.add(v)
        layers[i] = next_layer
        frontier = next_layer
        if not frontier:
            break
    return layers


def compute_ni_features(node, neighbors, max_layer):
    """Returns [ni_1, ni_2, ...]"""
    layers = bfs_layers_undirected(node, neighbors, max_hop=max_layer)
    feats = []
    for i in range(1, max_layer + 1):
        n_i = float(len(layers.get(i, set())))
        feats.append(n_i)
    return feats


def compute_Ii_features(node, neighbors, max_layer):
    """Returns [Ii_1, Ii_2, ...]"""
    layers = bfs_layers_undirected(node, neighbors, max_hop=max_layer)
    feats = []
    for i in range(1, max_layer + 1):
        layer_nodes = list(layers.get(i, set()))
        if len(layer_nodes) < 2:
            feats.append(0.0)
            continue
        if len(layer_nodes) > 100:
            import random
            random.seed(42)
            layer_nodes = random.sample(layer_nodes, 100)
        edge_count = 0
        for j, u in enumerate(layer_nodes):
            u_neighs = neighbors.get(str(u), set())
            for v in layer_nodes[j+1:]:
                if v in u_neighs:
                    edge_count += 1
        feats.append(float(edge_count))
    return feats


def compute_Ei_features(node, neighbors, max_layer):
    """Returns [Ei_1, Ei_2, ...]"""
    layers = bfs_layers_undirected(node, neighbors, max_hop=max_layer+1)
    feats = []
    for i in range(1, max_layer + 1):
        layer_i = layers.get(i, set())
        layer_ip1 = layers.get(i + 1, set())
        if not layer_i or not layer_ip1:
            feats.append(0.0)
            continue
        edge_count = 0
        layer_ip1_set = set(layer_ip1)
        for u in layer_i:
            u_neighs = neighbors.get(str(u), set())
            edge_count += len(u_neighs.intersection(layer_ip1_set))
        feats.append(float(edge_count))
    return feats


def extract_features(h, t, neighbors, feature_config, max_layer=3):
    """Extract pairwise features based on config."""
    feats = []
    
    # ni features
    if any(k.startswith('ni') for k in feature_config.keys() if feature_config[k]):
        ni_h = compute_ni_features(h, neighbors, max_layer)
        ni_t = compute_ni_features(t, neighbors, max_layer)
        
        if feature_config.get('ni_L1', False):
            feats.extend([ni_h[0], ni_t[0]])
        if feature_config.get('ni_L2', False) and len(ni_h) > 1:
            feats.extend([ni_h[1], ni_t[1]])
    
    # Ii features
    if any(k.startswith('Ii') for k in feature_config.keys() if feature_config[k]):
        Ii_h = compute_Ii_features(h, neighbors, max_layer)
        Ii_t = compute_Ii_features(t, neighbors, max_layer)
        
        if feature_config.get('Ii_L1', False):
            feats.extend([Ii_h[0], Ii_t[0]])
        if feature_config.get('Ii_L2', False) and len(Ii_h) > 1:
            feats.extend([Ii_h[1], Ii_t[1]])
    
    # Ei features
    if any(k.startswith('Ei') for k in feature_config.keys() if feature_config[k]):
        Ei_h = compute_Ei_features(h, neighbors, max_layer)
        Ei_t = compute_Ei_features(t, neighbors, max_layer)
        
        if feature_config.get('Ei_L1', False):
            feats.extend([Ei_h[0], Ei_t[0]])
        if feature_config.get('Ei_L2', False) and len(Ei_h) > 1:
            feats.extend([Ei_h[1], Ei_t[1]])
    
    return np.array(feats, dtype=np.float32)


def build_pairwise_dataset(df_triples, neighbors, a1h_scores, a1t_scores,
                          feature_config, a1h_thr=0.5, a1t_thr=0.5,
                          neg_per_pos=10, rng_seed=42):
    print(f"[+] Building pairwise dataset")
    rng = np.random.default_rng(rng_seed)
    
    cand_heads = [h for h, s in a1h_scores.items() if s >= a1h_thr]
    cand_tails = [t for t, s in a1t_scores.items() if s >= a1t_thr]
    print(f"    Candidate heads: {len(cand_heads):,}, tails: {len(cand_tails):,}")
    
    cand_heads_set = set(cand_heads)
    cand_tails_set = set(cand_tails)
    
    X_pairs, y_pairs, heads_list, tails_list = [], [], [], []
    tail_pool = np.array(cand_tails, dtype=object)
    
    for idx, row in df_triples.iterrows():
        h, t = str(row.head_id), str(row.tail_id)
        if h not in cand_heads_set or t not in cand_tails_set:
            continue
        if h not in neighbors or t not in neighbors:
            continue
        
        feats_pos = extract_features(h, t, neighbors, feature_config, max_layer=3)
        X_pairs.append(feats_pos)
        y_pairs.append(1)
        heads_list.append(h)
        tails_list.append(t)
        
        for _ in range(neg_per_pos):
            neg_t = str(rng.choice(tail_pool))
            if neg_t == t:
                continue
            feats_neg = extract_features(h, neg_t, neighbors, feature_config, max_layer=3)
            X_pairs.append(feats_neg)
            y_pairs.append(0)
            heads_list.append(h)
            tails_list.append(neg_t)
    
    X = np.array(X_pairs, dtype=np.float32)
    y = np.array(y_pairs, dtype=np.int64)
    heads = np.array(heads_list, dtype=object)
    tails = np.array(tails_list, dtype=object)
    
    print(f"    Dataset: {len(y):,} pairs (pos={y.sum():,}, neg={(y==0).sum():,})")
    print(f"    Features: {X.shape[1]}")
    
    return X, y, heads, tails


def compute_ranking_metrics(test_pairs_with_scores, ks=[1, 3, 5, 10]):
    """Compute MRR and Hits@k metrics."""
    from collections import defaultdict
    head_to_pairs = defaultdict(list)
    for h, t, label, score in test_pairs_with_scores:
        head_to_pairs[h].append((t, label, score))
    
    mrr_list, hits = [], {k: [] for k in ks}
    for h, pairs in head_to_pairs.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
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
    
    return {
        'mrr': float(np.mean(mrr_list)) if mrr_list else 0.0,
        'hits': {f'hits@{k}': float(np.mean(hits[k])) if hits[k] else 0.0 for k in ks},
        'n_heads': len(mrr_list),
    }


def train_and_evaluate_pytorch(X_train, y_train, heads_train, tails_train,
                               X_test, y_test, heads_test, tails_test,
                               device, batch_size=512, epochs=100):
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")
    
    model.eval()
    with torch.no_grad():
        y_scores_test = model(X_test_t).cpu().numpy().flatten()
    
    test_pairs = [(heads_test[i], tails_test[i], y_test[i], y_scores_test[i])
                  for i in range(len(y_test))]
    metrics = compute_ranking_metrics(test_pairs, ks=[1, 3, 5, 10])
    return metrics


def run_experiment(exp_name, feature_config, df_triples, neighbors,
                  a1h_scores, a1t_scores, device, outdir, seed=42,
                  a1h_thr=0.5, a1t_thr=0.5, neg_per_pos=10):
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}")
    
    X, y, heads, tails = build_pairwise_dataset(df_triples, neighbors,
                                                a1h_scores, a1t_scores, feature_config,
                                                a1h_thr=a1h_thr, a1t_thr=a1t_thr,
                                                neg_per_pos=neg_per_pos, rng_seed=seed)
    
    if len(y) < 100:
        print(f"⚠️  Too few samples ({len(y)}), skipping")
        return None
    
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    heads_train, heads_test = heads[train_idx], heads[test_idx]
    tails_train, tails_test = tails[train_idx], tails[test_idx]
    
    print(f"[+] Train: {len(y_train):,} | Test: {len(y_test):,}")
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    print(f"[+] Training on {device}")
    metrics = train_and_evaluate_pytorch(X_train_sc, y_train, heads_train, tails_train,
                                        X_test_sc, y_test, heads_test, tails_test,
                                        device, batch_size=512, epochs=100)
    
    mrr = metrics['mrr']
    hits_1 = metrics['hits']['hits@1']
    hits_3 = metrics['hits']['hits@3']
    hits_5 = metrics['hits']['hits@5']
    hits_10 = metrics['hits']['hits@10']
    
    print(f"\n✅ MRR = {mrr:.4f} | H@1 = {hits_1:.4f} | H@3 = {hits_3:.4f} | H@5 = {hits_5:.4f} | H@10 = {hits_10:.4f}")
    
    exp_dir = outdir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    save_metrics = {
        "experiment": exp_name,
        "feature_config": feature_config,
        "num_features": int(X.shape[1]),
        "mrr": mrr,
        "hits@1": hits_1,
        "hits@3": hits_3,
        "hits@5": hits_5,
        "hits@10": hits_10,
    }
    
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2)
    
    return save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-path", type=str, required=True)
    ap.add_argument("--public-has-header", action="store_true")
    ap.add_argument("--sens-path", type=str, required=True)
    ap.add_argument("--attack1-head-scores", type=str, required=True)
    ap.add_argument("--attack1-tail-scores", type=str, required=True)
    ap.add_argument("--a1h-thr", type=float, default=0.5)
    ap.add_argument("--a1t-thr", type=float, default=0.5)
    ap.add_argument("--neg-per-pos", type=int, default=10)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[+] Using {device}")
    
    public_path = Path(args.public_path).expanduser()
    sens_path = Path(args.sens_path).expanduser()
    a1h_path = Path(args.attack1_head_scores).expanduser()
    a1t_path = Path(args.attack1_tail_scores).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    
    neighbors, degrees = load_public_graph(public_path, args.public_has_header)
    df_triples = load_sensitive_triples(sens_path)
    
    print(f"\n[+] Loading Attack1 scores")
    a1h_scores = load_attack1_scores(a1h_path, id_col="head_id", score_col="score")
    a1t_scores = load_attack1_scores(a1t_path, id_col="tail_id", score_col="score")
    print(f"    Head scores: {len(a1h_scores):,}, Tail scores: {len(a1t_scores):,}")
    
    # 14 LOGICAL PAIRWISE EXPERIMENTS
    experiments = [
        # LAYER 1 (7 experiments)
        {'name': '01_ni_L1', 'config': {'ni_L1': True}},
        {'name': '02_Ii_L1', 'config': {'Ii_L1': True}},
        {'name': '03_Ei_L1', 'config': {'Ei_L1': True}},
        {'name': '04_ni+Ii_L1', 'config': {'ni_L1': True, 'Ii_L1': True}},
        {'name': '05_ni+Ei_L1', 'config': {'ni_L1': True, 'Ei_L1': True}},
        {'name': '06_Ii+Ei_L1', 'config': {'Ii_L1': True, 'Ei_L1': True}},
        {'name': '07_ALL_L1', 'config': {'ni_L1': True, 'Ii_L1': True, 'Ei_L1': True}},
        
        # LAYER 2 (7 experiments) - CUMULATIVE
        {'name': '08_ni_L2', 'config': {'ni_L1': True, 'ni_L2': True}},
        {'name': '09_Ii_L2', 'config': {'Ii_L1': True, 'Ii_L2': True}},
        {'name': '10_Ei_L2', 'config': {'Ei_L1': True, 'Ei_L2': True}},
        {'name': '11_ni+Ii_L2', 'config': {'ni_L1': True, 'ni_L2': True, 'Ii_L1': True, 'Ii_L2': True}},
        {'name': '12_ni+Ei_L2', 'config': {'ni_L1': True, 'ni_L2': True, 'Ei_L1': True, 'Ei_L2': True}},
        {'name': '13_Ii+Ei_L2', 'config': {'Ii_L1': True, 'Ii_L2': True, 'Ei_L1': True, 'Ei_L2': True}},
        {'name': '14_ALL_L2', 'config': {'ni_L1': True, 'ni_L2': True, 'Ii_L1': True, 'Ii_L2': True, 'Ei_L1': True, 'Ei_L2': True}},
    ]
    
    results = []
    for exp in experiments:
        result = run_experiment(
            exp['name'], exp['config'], df_triples, neighbors,
            a1h_scores, a1t_scores, device, outdir, args.seed,
            a1h_thr=args.a1h_thr, a1t_thr=args.a1t_thr,
            neg_per_pos=args.neg_per_pos
        )
        if result is not None:
            results.append(result)
    
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    for _, row in df_results.iterrows():
        print(f"{row['experiment']:20s} | Feats: {row['num_features']:2d} | "
              f"MRR: {row['mrr']:.4f} | H@1: {row['hits@1']:.4f} | "
              f"H@3: {row['hits@3']:.4f} | H@5: {row['hits@5']:.4f} | H@10: {row['hits@10']:.4f}")
    
    df_results.to_csv(outdir / "summary_ablation.tsv", sep="\t", index=False)
    print(f"\n✅ Summary: {outdir / 'summary_ablation.tsv'}")


if __name__ == "__main__":
    main()