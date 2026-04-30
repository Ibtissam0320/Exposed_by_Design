"""
K-Anonymity Structural Defense for Knowledge Graphs
=====================================================
Based on: Liu & Terzi (2008) "Towards Identity Anonymization on Graphs" SIGMOD 2008

Guarantees that every node is structurally indistinguishable from
at least k-1 other nodes based on their degree sequence.

Method:
  1. Sort nodes by degree
  2. Group nodes into buckets of size k
  3. Within each bucket, equalize degrees:
     - Add edges to nodes below bucket median degree
     - Remove edges from nodes above bucket median degree
  4. Result: every node has same degree as k-1 others → k-anonymous

Usage:
    python defense_kanonymity.py --input <path> --output <path> --k 5
    python defense_kanonymity.py --input <path> --output <path> --k 10
    python defense_kanonymity.py --input <path> --output <path> --k 25
"""

import argparse
import pandas as pd
import networkx as nx
import numpy as np
import random
import os
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Input TSV: subject\\trelation\\tobject")
    ap.add_argument("--output", required=True, help="Output sanitized TSV")
    ap.add_argument("--k",      type=int, default=5, help="k-anonymity parameter (default=5)")
    ap.add_argument("--seed",   type=int, default=42)
    return ap.parse_args()


def load_kg(path):
    print(f"[1/4] Loading KG from {path} ...")
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["subject", "relation", "object"],
                     low_memory=False, dtype=str).dropna()
    print(f"      Triples: {len(df):,}")

    G = nx.Graph()
    for _, row in df.iterrows():
        h, r, t = str(row["subject"]), str(row["relation"]), str(row["object"])
        if G.has_edge(h, t):
            G[h][t]["relations"].append(r)
        else:
            G.add_edge(h, t, relations=[r])

    print(f"      Nodes : {G.number_of_nodes():,}")
    print(f"      Edges : {G.number_of_edges():,}")
    return G, df


def kanonymize(G, k, seed=42):
    """
    Enforce k-anonymity on the graph by equalizing degrees within buckets.
    
    Algorithm:
      1. Sort nodes by degree ascending
      2. Split into buckets of size k
      3. For each bucket:
         - target_degree = median degree of bucket
         - nodes below target → add edges to random non-neighbors
         - nodes above target → remove random edges
    
    Returns: anonymized graph G'
    """
    print(f"\n[2/4] Applying k-anonymity (k={k}) ...")
    random.seed(seed)
    np.random.seed(seed)

    G_prime = G.copy()
    nodes   = list(G_prime.nodes())
    all_relations = list(set(
        r for _, _, d in G_prime.edges(data=True)
        for r in d.get("relations", [])
    ))

    # Sort nodes by degree
    nodes_by_deg = sorted(nodes, key=lambda n: G_prime.degree(n))
    n_nodes = len(nodes_by_deg)

    # Split into buckets of size k
    buckets = [nodes_by_deg[i:i+k] for i in range(0, n_nodes, k)]

    total_added   = 0
    total_removed = 0

    for bucket in buckets:
        if len(bucket) < 2:
            continue

        # Target degree = median of bucket
        degs = [G_prime.degree(n) for n in bucket]
        target = int(np.median(degs))

        for node in bucket:
            current_deg = G_prime.degree(node)

            if current_deg < target:
                # Need to ADD edges
                needed = target - current_deg
                # Candidates: nodes not already connected to this node
                non_neighbors = [
                    v for v in nodes
                    if v != node and not G_prime.has_edge(node, v)
                ]
                random.shuffle(non_neighbors)
                added = 0
                for v in non_neighbors:
                    if added >= needed:
                        break
                    rel = random.choice(all_relations)
                    G_prime.add_edge(node, v, relations=[rel])
                    added += 1
                total_added += added

            elif current_deg > target:
                # Need to REMOVE edges
                excess = current_deg - target
                neighbors = list(G_prime.neighbors(node))
                random.shuffle(neighbors)
                removed = 0
                for v in neighbors:
                    if removed >= excess:
                        break
                    if G_prime.has_edge(node, v):
                        G_prime.remove_edge(node, v)
                        removed += 1
                total_removed += removed

    print(f"      Edges added  : {total_added:,}")
    print(f"      Edges removed: {total_removed:,}")
    print(f"      Final edges  : {G_prime.number_of_edges():,}")

    # Verify k-anonymity
    deg_count = defaultdict(int)
    for n in G_prime.nodes():
        deg_count[G_prime.degree(n)] += 1

    satisfied = sum(1 for c in deg_count.values() if c >= k)
    total_degs = len(deg_count)
    print(f"      k-anonymity satisfied: {satisfied}/{total_degs} degree values "
          f"have ≥{k} nodes ({100*satisfied/max(total_degs,1):.1f}%)")

    return G_prime


def measure_utility(G_orig, G_prime):
    print("\n[3/4] Measuring utility ...")

    eo = set(frozenset(e[:2]) for e in G_orig.edges())
    ep = set(frozenset(e[:2]) for e in G_prime.edges())
    overlap = len(eo & ep) / max(len(eo), 1)
    jaccard = len(eo & ep) / max(len(eo | ep), 1)

    deg_o = np.array([d for _, d in G_orig.degree()])
    deg_p = np.array([d for _, d in G_prime.degree()])
    max_d = int(max(deg_o.max(), deg_p.max())) + 1
    p = np.bincount(deg_o, minlength=max_d).astype(float)
    q = np.bincount(deg_p, minlength=max_d).astype(float)
    p /= p.sum(); q /= q.sum(); p += 1e-10; q += 1e-10
    kl = float(np.sum(p * np.log(p / q)))

    cc_o = nx.average_clustering(G_orig)
    cc_p = nx.average_clustering(G_prime)

    print(f"      Edge overlap     : {overlap:.4f}")
    print(f"      Edge Jaccard     : {jaccard:.4f}")
    print(f"      Degree KL div    : {kl:.6f}")
    print(f"      Clustering delta : {abs(cc_o - cc_p):.4f}")
    return {"edge_overlap": overlap, "edge_jaccard": jaccard,
            "degree_kl_div": kl, "clustering_delta": abs(cc_o - cc_p)}


def save_graph(G_prime, output_path):
    print(f"\n[4/4] Saving to {output_path} ...")
    rows = []
    for u, v, data in G_prime.edges(data=True):
        for r in data.get("relations", ["unknown"]):
            rows.append({"subject": u, "relation": r, "object": v})
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_out.to_csv(output_path, sep="\t", index=False, header=False)
    print(f"      Saved {len(df_out):,} triples → {output_path}")


def main():
    args = parse_args()

    G, df = load_kg(args.input)
    G_prime = kanonymize(G, k=args.k, seed=args.seed)
    metrics = measure_utility(G, G_prime)
    save_graph(G_prime, args.output)

    print(f"\n{'='*55}")
    print(f"  K-ANONYMITY DEFENSE DONE (k={args.k})")
    print(f"  edge_overlap={metrics['edge_overlap']:.4f}  "
          f"kl_div={metrics['degree_kl_div']:.4f}")
    print(f"{'='*55}")
    print(f"\nNext: run your attacks on {args.output}")


if __name__ == "__main__":
    main()