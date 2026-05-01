"""
CHAMELEON — Dynamic Defense
=========================================================
Detects the structural profile of sensitive nodes once,
then selects the appropriate defense for all budgets:



Usage:
  python dynamic_defense.py \\
    --public-path  /path/to/global_kg_public.tsv \\
    --sens-dir     /path/to/sensitive/ \\
    --outdir       /path/to/output/ \\
    --budgets      0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \\
    --seed         42
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────

def load_graph_nx(path):
    """Load as NetworkX MultiDiGraph (used by TFE)."""
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['h', 'r', 't'], dtype=str).dropna()
    G = nx.MultiDiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['h'], row['t'], relation=row['r'], fake=False)
    return G


def load_graph_prism(path):
    """Load as undirected Graph with relation lists (used by PRISM kg_defense_v2 style)."""
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['h', 'r', 't'], dtype=str).dropna()
    G = nx.Graph()
    for _, row in df.iterrows():
        u, r, v = row['h'], row['r'], row['t']
        if G.has_edge(u, v):
            G[u][v]['relations'].append(r)
        else:
            G.add_edge(u, v, relations=[r])
    return G, df


def load_all_sensitive_heads(sens_dir):
    all_heads = set()
    for fname in os.listdir(sens_dir):
        if not fname.endswith('.tsv'):
            continue
        fpath = os.path.join(sens_dir, fname)
        try:
            df = pd.read_csv(fpath, sep='\t', header=None,
                             names=['h', 'r', 't'], dtype=str).dropna()
            all_heads |= set(df['h'].unique())
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")
    print(f"  Total unique sensitive heads: {len(all_heads):,}")
    return all_heads


def save_graph_nx(G, path):
    rows = [{'h': u, 'r': d['relation'], 't': v}
            for u, v, d in G.edges(data=True)]
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False, header=False)
    print(f"  [Save] {len(rows):,} triples → {path}")


def save_graph_prism(G, path):
    rows = []
    for u, v, data in G.edges(data=True):
        for r in data.get('relations', ['unknown']):
            rows.append({'h': u, 'r': r, 't': v})
    pd.DataFrame(rows).to_csv(path, sep='\t', index=False, header=False)
    print(f"  [Save] {len(rows):,} triples → {path}")

# ─────────────────────────────────────────────────────────────────
# 2. PROFILE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_profile(G_nx, sensitive_heads):
    """
    Compute μ_S vs μ_NS for all 4 structural features.
    A feature is INVERTED if μ_S < μ_NS.

    Decision rule:
      strictly more than 2 features inverted (>2/4) → TFE v2
      otherwise                                      → PRISM
    """
    s_nodes  = [n for n in sensitive_heads
                if n in G_nx.nodes() and G_nx.out_degree(n) > 0]
    ns_nodes = [n for n in G_nx.nodes()
                if n not in sensitive_heads and G_nx.out_degree(n) > 0]

    if not s_nodes or not ns_nodes:
        print("  [Profile] Not enough nodes — defaulting to PRISM")
        return 'normal'

    def feat_stats(nodes):
        out_degs    = [G_nx.out_degree(n) for n in nodes]
        in_degs     = [G_nx.in_degree(n)  for n in nodes]
        rel_divs    = [len(set(d['relation']
                        for _, _, d in G_nx.out_edges(n, data=True)))
                       for n in nodes]
        avg_nb_degs = []
        for n in nodes:
            nbs = list(G_nx.successors(n)) + list(G_nx.predecessors(n))
            avg_nb_degs.append(
                float(np.mean([G_nx.out_degree(nb) + G_nx.in_degree(nb)
                               for nb in nbs])) if nbs else 0.0)
        return {
            'out_deg':    float(np.median(out_degs)),
            'in_deg':     float(np.median(in_degs)),
            'rel_div':    float(np.median(rel_divs)),
            'avg_nb_deg': float(np.median(avg_nb_degs)),
        }

    s_med  = feat_stats(s_nodes)
    ns_med = feat_stats(ns_nodes)

    print(f"\n  [CHAMELEON — Profile Detection]")
    print(f"  {'Feature':<15} {'μ_S':>8} {'μ_NS':>8} {'Inverted?':>12}")
    print(f"  {'─'*47}")

    n_inverted = 0
    for feat in ['out_deg', 'in_deg', 'rel_div', 'avg_nb_deg']:
        inv = (s_med[feat] > ns_med[feat]) and (s_med[feat] / max(ns_med[feat], 0.01) > 2.0)
        if inv:
            n_inverted += 1
        print(f"  {feat:<15} {s_med[feat]:>8.2f} {ns_med[feat]:>8.2f} "
              f"{'❌ INV' if inv else '✅':>12}")

    print(f"\n  Inverted features : {n_inverted}/4")

    if n_inverted > 2:
        print(f"  → INVERTED profile (>{2}/4) → TFE v2")
        return 'inverted'
    else:
        print(f"  → NORMAL profile (≤2/4 inverted) → PRISM")
        return 'normal'

# ─────────────────────────────────────────────────────────────────
# 3. PRISM — kg_defense_v2 (exact, inlined)
# ─────────────────────────────────────────────────────────────────

def _load_kg_prism(path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["h", "r", "t"], dtype=str,
                     low_memory=False).dropna()
    df = df.map(str.strip)
    G  = nx.Graph()
    for _, row in df.iterrows():
        u, r, v = row["h"], row["r"], row["t"]
        if G.has_edge(u, v):
            G[u][v]["relations"].append(r)
        else:
            G.add_edge(u, v, relations=[r])
    return G, df["r"].unique().tolist()


def _compute_risk_scores(G):
    edges = list(G.edges())
    deg   = dict(G.degree())
    clust = nx.clustering(G)
    deg_count = Counter(deg.values())
    k_anon    = {n: deg_count[deg[n]] for n in G.nodes()}
    max_k     = max(k_anon.values())
    max_dp    = max(deg[u] * deg[v] for u, v in edges) or 1
    risk     = {}
    cn_cache = {}
    for u, v in edges:
        cn = len(list(nx.common_neighbors(G, u, v)))
        cn_cache[(u, v)] = cn
        cn_norm   = cn / max(1, min(deg[u], deg[v]))
        kanon_r   = 1.0 - min(k_anon[u], k_anon[v]) / max_k
        bet_proxy = (deg[u] * deg[v]) / max_dp
        clust_avg = (clust[u] + clust[v]) / 2.0
        risk[(u, v)] = (0.30 * cn_norm + 0.25 * kanon_r +
                        0.25 * bet_proxy + 0.20 * clust_avg)
    sorted_edges = sorted(risk.items(), key=lambda x: x[1], reverse=True)
    return sorted_edges, cn_cache


def _triangle_disruption_delete(G, n_delete, cn_cache):
    edges_by_cn = sorted(cn_cache.items(), key=lambda x: x[1], reverse=True)
    deleted = 0
    for (u, v), cn in edges_by_cn:
        if deleted >= n_delete:
            break
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            deleted += 1
    return G


def _add_plausible_fake_edges(G, n_add, all_relations):
    nodes    = list(G.nodes())
    added    = 0
    attempts = 0
    while added < n_add and attempts < n_add * 50:
        attempts += 1
        u = random.choice(nodes)
        nbs_u = list(G.neighbors(u))
        if not nbs_u:
            continue
        w = random.choice(nbs_u)
        nbs_w = list(G.neighbors(w))
        if not nbs_w:
            continue
        v = random.choice(nbs_w)
        if v == u or G.has_edge(u, v):
            continue
        cn = len(list(nx.common_neighbors(G, u, v)))
        if 1 <= cn <= 3:
            G.add_edge(u, v, relations=[random.choice(all_relations)], fake=True)
            added += 1
    return G


def _obfuscate_node_relation_profiles(G, budget_fraction=0.15):
    node_rel = defaultdict(Counter)
    for u, v, data in G.edges(data=True):
        for r in data.get("relations", []):
            node_rel[u][r] += 1
            node_rel[v][r] += 1
    sigs      = {n: frozenset(p.keys()) for n, p in node_rel.items()}
    sig_count = Counter(sigs.values())
    uniqueness = {n: 1.0 / sig_count[s] for n, s in sigs.items()}
    sorted_nodes = sorted(uniqueness, key=uniqueness.get, reverse=True)
    target_nodes = set(sorted_nodes[:int(len(sorted_nodes) * budget_fraction)])
    all_relations = list(set(r for _, _, d in G.edges(data=True)
                             for r in d.get("relations", [])))
    for u, v, data in G.edges(data=True):
        if (u in target_nodes or v in target_nodes) and data.get("relations"):
            if random.random() < 0.3:
                rels = data["relations"][:]
                rels[random.randint(0, len(rels)-1)] = random.choice(all_relations)
                G[u][v]["relations"] = rels
    return G


def _rewire_edges(G, sorted_edges, n_rewire):
    nodes   = list(G.nodes())
    mid     = len(sorted_edges) // 4
    rewired = 0
    for (u, v), _ in sorted_edges[mid:]:
        if rewired >= n_rewire:
            break
        if G.has_edge(u, v):
            rels  = G[u][v]["relations"][:]
            new_v = random.choice(nodes)
            if new_v != u and not G.has_edge(u, new_v):
                G.remove_edge(u, v)
                G.add_edge(u, new_v, relations=rels)
                rewired += 1
    return G



def _targeted_delete_S_edges(G, n_delete, sensitive_heads):
    """
    Delete edges incident to S nodes to bring their degree
    toward μ_NS. Stops when S degree reaches μ_NS.
    """
    s_nodes  = [n for n in sensitive_heads if n in G.nodes()]
    ns_nodes = [n for n in G.nodes() if n not in sensitive_heads]
    if not s_nodes or not ns_nodes:
        return G, 0

    mu_ns = float(np.median([G.degree(n) for n in ns_nodes]))

    # Collect S edges sorted by node degree descending (greedily reduce densest S first)
    s_edges = []
    for u, v in G.edges():
        if u in sensitive_heads or v in sensitive_heads:
            s_edges.append((u, v))
    s_edges.sort(key=lambda e: max(G.degree(e[0]), G.degree(e[1])), reverse=True)

    deleted = 0
    for u, v in s_edges:
        if deleted >= n_delete:
            break
        if not G.has_edge(u, v):
            continue
        # Only delete if the S endpoint is still above μ_NS
        s_endpoint = u if u in sensitive_heads else v
        if G.degree(s_endpoint) <= mu_ns:
            continue
        G.remove_edge(u, v)
        deleted += 1

    # Fallback if not enough
    if deleted < n_delete:
        G = _triangle_disruption_delete(G, n_delete - deleted, {})

    return G, deleted


def _targeted_add_NS_edges(G, n_add, all_relations, sensitive_heads):
    """
    Add fake edges to NS nodes to bring their degree toward μ_S.
    Stops adding to a node once it reaches μ_S.
    """
    s_nodes  = [n for n in sensitive_heads if n in G.nodes()]
    ns_nodes = [n for n in G.nodes() if n not in sensitive_heads]
    if not ns_nodes:
        return G, 0

    mu_s = float(np.median([G.degree(n) for n in s_nodes])) if s_nodes else 10.0

    added    = 0
    attempts = 0
    # Prioritize NS nodes furthest below μ_S
    ns_sorted = sorted(ns_nodes, key=lambda n: G.degree(n))

    while added < n_add and attempts < n_add * 100:
        attempts += 1
        # Pick NS node that is still below μ_S
        u = random.choice(ns_sorted[:max(1, len(ns_sorted)//2)])
        if G.degree(u) >= mu_s:
            continue
        v = random.choice(ns_nodes)
        if v == u or G.has_edge(u, v):
            continue
        G.add_edge(u, v, relations=[random.choice(all_relations)], fake=True)
        added += 1

    return G, added

def prism_sanitize(outdir, budget, seed,
                   G_orig, all_relations, sorted_edges, cn_cache,
                   sensitive_heads=None):
    """
    PRISM = kg_defense_v2 — semi-targeted.
    Deletes edges from S nodes, adds fake edges to NS nodes.
    G_orig, all_relations, sorted_edges, cn_cache preloaded once outside loop.
    """
    bpct     = int(budget * 100)
    out_path = os.path.join(outdir,
                            f"kg_chameleon_prism_budget_{bpct:02d}pct.tsv")

    random.seed(seed)
    np.random.seed(seed)

    G        = G_orig.copy()
    n_edges  = G.number_of_edges()
    n_budget = int(n_edges * budget)
    n_delete = int(n_budget * 0.50)
    n_rewire = int(n_budget * 0.20)
    # Add back deleted edges + original add budget → preserve |E|
    n_add    = n_delete + (n_budget - n_delete - n_rewire)

    print(f"  [PRISM] budget={budget:.0%} → "
          f"delete={n_delete}, rewire={n_rewire}, add={n_add} "
          f"(edge count preserved: {n_edges:,})")

    if sensitive_heads:
        G, deleted = _targeted_delete_S_edges(G, n_delete, sensitive_heads)
        print(f"  [PRISM] Targeted delete: {deleted:,} S-incident edges removed")
    else:
        G = _triangle_disruption_delete(G, n_delete, cn_cache.copy())

    G = _rewire_edges(G, sorted_edges, n_rewire)

    if sensitive_heads:
        G, added = _targeted_add_NS_edges(G, n_add, all_relations, sensitive_heads)
        print(f"  [PRISM] Targeted add: {added:,} fake edges on NS nodes")
    else:
        G = _add_plausible_fake_edges(G, n_add, all_relations)

    G = _obfuscate_node_relation_profiles(G, budget_fraction=min(budget, 0.30))

    rows = []
    for u, v, data in G.edges(data=True):
        for r in data.get("relations", ["unknown"]):
            rows.append([u, r, v])
    pd.DataFrame(rows, columns=["h", "r", "t"]) \
      .to_csv(out_path, sep="\t", index=False, header=False)
    print(f"  [Save] {len(rows):,} triples → {out_path}")
    return out_path


# 4. TFE v2 — (inverted profile)
# ─────────────────────────────────────────────────────────────────

def compute_features_nx(G):
    features = {}
    for node in G.nodes():
        out_edges  = list(G.out_edges(node, data=True))
        in_edges   = list(G.in_edges(node, data=True))
        out_deg    = len(out_edges)
        in_deg     = len(in_edges)
        rel_div    = len(set(d['relation'] for _, _, d in out_edges)) if out_edges else 0
        neighbors  = list(G.successors(node)) + list(G.predecessors(node))
        avg_nb_deg = float(np.mean([G.out_degree(n) + G.in_degree(n)
                                    for n in neighbors])) if neighbors else 0.0
        features[node] = {
            'out_deg': out_deg, 'in_deg': in_deg,
            'rel_div': rel_div, 'avg_nb_deg': avg_nb_deg
        }
    return features


def compute_distributions_nx(features, sensitive_heads, G):
    s_nodes  = [n for n in sensitive_heads
                if n in G.nodes() and G.out_degree(n) > 0]
    ns_nodes = [n for n in G.nodes()
                if n not in sensitive_heads and G.out_degree(n) > 0]

    def stats(nodes, feat):
        vals = [features[n][feat] for n in nodes if n in features]
        if not vals:
            return {'median': 1.0, 'std': 0.5, 'min': 0.0, 'max': 10.0}
        return {
            'median': float(np.median(vals)),
            'std':    max(float(np.std(vals)), 0.5),
            'min':    float(np.min(vals)),
            'max':    float(np.max(vals))
        }

    feats    = ['out_deg', 'in_deg', 'rel_div', 'avg_nb_deg']
    s_stats  = {f: stats(s_nodes,  f) for f in feats}
    ns_stats = {f: stats(ns_nodes, f) for f in feats}
    return s_stats, ns_stats, ns_nodes, s_nodes


def sample_lognormal(stats_dict, feat, np_rng):
    s = stats_dict[feat]
    if s['median'] > 0:
        mu_ln    = np.log(s['median'] + 1)
        sigma_ln = min(s['std'] / (s['median'] + 1), 1.5)
        sample   = np.exp(np_rng.normal(mu_ln, sigma_ln)) - 1
    else:
        sample = np_rng.exponential(scale=max(s['std'], 1.0))
    return max(0, int(round(np.clip(sample, s['min'], s['max']))))


def adjust_out_deg(G, node, target, all_rel, rng, np_rng, prefer_high=False):
    n_ops = 0
    cur   = G.out_degree(node)
    if cur > target:
        edges = list(G.out_edges(node, keys=True, data=True))
        edges.sort(key=lambda x: (not x[3].get('fake', False), rng.random()))
        for u, v, k, _ in edges[:cur - target]:
            G.remove_edge(u, v, k)
            n_ops += 1
    elif cur < target:
        nodes = list(G.nodes())
        cands = (sorted(nodes, key=lambda n: G.out_degree(n)+G.in_degree(n),
                        reverse=True) if prefer_high else nodes)
        added, attempts = 0, 0
        while added < target - cur and attempts < (target-cur) * 150:
            attempts += 1
            if prefer_high and rng.random() < 0.7:
                v = cands[rng.randint(0, min(100, len(cands)-1))]
            else:
                nbs = list(G.successors(node))
                if nbs:
                    w    = rng.choice(nbs)
                    hop2 = list(G.successors(w))
                    v    = rng.choice(hop2) if hop2 else rng.choice(nodes)
                else:
                    v = rng.choice(nodes)
            if v == node or G.has_edge(node, v):
                continue
            G.add_edge(node, v, relation=rng.choice(all_rel), fake=True)
            added += 1
            n_ops += 1
    return n_ops


def adjust_in_deg(G, node, target, all_rel, rng):
    n_ops = 0
    cur   = G.in_degree(node)
    if cur > target:
        edges = list(G.in_edges(node, keys=True, data=True))
        edges.sort(key=lambda x: (not x[3].get('fake', False), rng.random()))
        for u, v, k, _ in edges[:cur - target]:
            G.remove_edge(u, v, k)
            n_ops += 1
    elif cur < target:
        nodes = list(G.nodes())
        added, attempts = 0, 0
        while added < target - cur and attempts < (target-cur) * 100:
            attempts += 1
            u = rng.choice(nodes)
            if u == node or G.has_edge(u, node):
                continue
            G.add_edge(u, node, relation=rng.choice(all_rel), fake=True)
            added += 1
            n_ops += 1
    return n_ops


def adjust_rel_div(G, node, target, all_rel, rng):
    out_edges    = list(G.out_edges(node, data=True))
    current_rels = set(d['relation'] for _, _, d in out_edges)
    n_ops        = 0
    if len(current_rels) < target:
        unused = [r for r in all_rel if r not in current_rels]
        rng.shuffle(unused)
        nodes = list(G.nodes())
        for rel in unused[:target - len(current_rels)]:
            for _ in range(50):
                v = rng.choice(nodes)
                if v != node and not G.has_edge(node, v):
                    G.add_edge(node, v, relation=rel, fake=True)
                    n_ops += 1
                    break
    elif len(current_rels) > target:
        for rel in list(current_rels)[target:]:
            for u, v, k, d in list(G.out_edges(node, keys=True, data=True)):
                if d['relation'] == rel and d.get('fake', False):
                    G.remove_edge(u, v, k)
                    n_ops += 1
                    break
    return n_ops


def find_ns_in_radius(features, ns_nodes, s_stats):
    return [n for n in ns_nodes
            if all(abs(features.get(n, {}).get(f, 0) - s_stats[f]['median'])
                   <= s_stats[f]['std']
                   for f in ['out_deg', 'in_deg', 'rel_div'])]


def tfe_sanitize(G_orig, sensitive_heads, budget, seed=42):
    print(f"  [TFE v2] budget={budget:.0%}")
    rng    = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    G      = G_orig.copy()

    all_rel = list(set(d['relation'] for _, _, d in G.edges(data=True)
                       if not d.get('fake', False))) or ['UNKNOWN']

    features = compute_features_nx(G)
    s_stats, ns_stats, ns_nodes, s_nodes = compute_distributions_nx(
        features, sensitive_heads, G)

    # Phase 1: S → NS
    def disc(node):
        f = features.get(node, {})
        return sum(abs(f.get(ft, 0) - ns_stats[ft]['median'])
                   / (ns_stats[ft]['std'] + 1e-6)
                   for ft in ['out_deg', 'in_deg', 'rel_div', 'avg_nb_deg'])

    s_sorted   = sorted(s_nodes, key=disc, reverse=True)
    n_eq       = max(1, int(budget * len(s_sorted)))
    total_ops  = 0

    print(f"  [Phase 1] S→NS: {n_eq}/{len(s_nodes)} nodes")
    for node in s_sorted[:n_eq]:
        total_ops += adjust_out_deg(G, node,
                         sample_lognormal(ns_stats, 'out_deg', np_rng),
                         all_rel, rng, np_rng, prefer_high=True)
        total_ops += adjust_in_deg(G, node,
                         sample_lognormal(ns_stats, 'in_deg', np_rng),
                         all_rel, rng)
        total_ops += adjust_rel_div(G, node,
                         sample_lognormal(ns_stats, 'rel_div', np_rng),
                         all_rel, rng)

    # Phase 2: NS → S aggressive
    features = compute_features_nx(G)
    s_stats, ns_stats, ns_nodes, s_nodes = compute_distributions_nx(
        features, sensitive_heads, G)
    ns_in_r = find_ns_in_radius(features, ns_nodes, s_stats)

    print(f"  [Phase 2] NS→S: {len(ns_in_r):,}/{len(ns_nodes):,} nodes")
    for node in ns_in_r:
        total_ops += adjust_out_deg(G, node,
                         sample_lognormal(s_stats, 'out_deg', np_rng),
                         all_rel, rng, np_rng, prefer_high=False)
        total_ops += adjust_in_deg(G, node,
                         sample_lognormal(s_stats, 'in_deg', np_rng),
                         all_rel, rng)
        total_ops += adjust_rel_div(G, node,
                         sample_lognormal(s_stats, 'rel_div', np_rng),
                         all_rel, rng)

    print(f"  [TFE v2] Total ops: {total_ops:,} | "
          f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

# ─────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CHAMELEON — Dynamic Defense')
    parser.add_argument('--public-path',  required=True)
    parser.add_argument('--sens-dir',     required=True)
    parser.add_argument('--outdir',       required=True)
    parser.add_argument('--budgets',      nargs='+', type=float,
                        default=[0.05, 0.10, 0.20, 0.30, 0.40,
                                 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument('--seed',         type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  CHAMELEON — Dynamic Defense")
    print(f"{'='*60}")
    print(f"  Public graph : {args.public_path}")
    print(f"  Sensitive dir: {args.sens_dir}")
    print(f"  Output dir   : {args.outdir}")
    print(f"  Budgets      : {args.budgets}")

    print("\n[+] Loading graph...")
    G_nx = load_graph_nx(args.public_path)
    print(f"    Nodes: {G_nx.number_of_nodes():,}  "
          f"Edges: {G_nx.number_of_edges():,}")

    print("\n[+] Loading sensitive heads...")
    sensitive_heads = load_all_sensitive_heads(args.sens_dir)

    # ── PROFILE DETECTION ────────────────────────────────────────
    profile      = detect_profile(G_nx, sensitive_heads)
    defense_name = "TFE v2" if profile == 'inverted' else "PRISM"

    print(f"\n{'─'*60}")
    print(f"  Selected defense : {defense_name}")
    print(f"{'─'*60}")

    # ── PRE-LOAD ONCE FOR PRISM (avoid reloading per budget) ─────
    if profile == 'normal':
        print("\n[+] Pre-loading graph data for PRISM (done once)...")
        G_prism, all_relations_p = _load_kg_prism(args.public_path)
        sorted_edges_p, cn_cache_p = _compute_risk_scores(G_prism)
        print(f"    Preload done — {G_prism.number_of_nodes():,} nodes, "
              f"{G_prism.number_of_edges():,} edges, "
              f"{len(cn_cache_p):,} risk scores")

    # ── RUN PER BUDGET ───────────────────────────────────────────
    for budget in args.budgets:
        print(f"\n{'─'*55}")
        print(f"  Budget = {budget:.0%}")

        if profile == 'inverted':
            G_san    = tfe_sanitize(G_nx, sensitive_heads,
                                    budget=budget, seed=args.seed)
            bpct     = int(budget * 100)
            out_path = os.path.join(args.outdir,
                                    f"kg_chameleon_tfe_budget_{bpct:02d}pct.tsv")
            save_graph_nx(G_san, out_path)
        else:
            prism_sanitize(
                outdir          = args.outdir,
                budget          = budget,
                seed            = args.seed,
                G_orig          = G_prism,
                all_relations   = all_relations_p,
                sorted_edges    = sorted_edges_p,
                cn_cache        = cn_cache_p,
                sensitive_heads = sensitive_heads,
            )

    print(f"\n{'='*60}")
    print(f"  CHAMELEON Done — Defense used: {defense_name}")
    print(f"  Output: {args.outdir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()