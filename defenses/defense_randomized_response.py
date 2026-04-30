"""
Randomized Response Defense for Knowledge Graphs
=================================================
Based on: Qin et al. (2017) "Generating Synthetic Decentralized Social Graphs
          with Local Differential Privacy" CCS 2017

Each node locally randomizes its adjacency list:
  - Keep existing edge   with probability p = e^epsilon / (e^epsilon + 1)
  - Add non-existing edge with probability q = 1 / (e^epsilon + 1)

This provides edge-level Local Differential Privacy with budget epsilon.
Purely structural — no embeddings needed.

Usage:
    python defense_randomized_response.py --input <path> --output <path> --epsilon 1.0
    python defense_randomized_response.py --input <path> --output <path> --epsilon 0.5
    python defense_randomized_response.py --input <path> --output <path> --epsilon 2.0

Epsilon guide:
    epsilon = 0.1  → very strong privacy, low utility
    epsilon = 0.5  → strong privacy
    epsilon = 1.0  → balanced (recommended start)
    epsilon = 2.0  → moderate privacy
    epsilon = 5.0  → weak privacy, high utility
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
    ap.add_argument("--input",   required=True,  help="Input TSV: subject\\trelation\\tobject")
    ap.add_argument("--output",  required=True,  help="Output sanitized TSV")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="Privacy budget epsilon (smaller=more private, default=1.0)")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--sample_nonedges", type=float, default=1.0,
                    help="Fraction of non-edges to consider (use <1.0 for large graphs)")
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


def randomized_response(G, epsilon, seed=42, sample_nonedges=1.0):
    """
    Apply Randomized Response with Local Differential Privacy.

    For each possible edge (u,v):
      - If edge EXISTS   : keep with prob p = e^ε / (e^ε + 1)
      - If edge NOT EXIST: add  with prob q = 1 / (e^ε + 1)

    This guarantees ε-edge Local Differential Privacy.
    (Qin et al. CCS 2017, Section 3)
    """
    print(f"\n[2/4] Applying Randomized Response (epsilon={epsilon}) ...")
    print(f"      p(keep edge)   = {np.exp(epsilon)/(np.exp(epsilon)+1):.4f}")
    print(f"      p(add nonedge) = {1/(np.exp(epsilon)+1):.4f}")

    random.seed(seed)
    np.random.seed(seed)

    G_prime = nx.Graph()
    G_prime.add_nodes_from(G.nodes())

    all_relations = list(set(
        r for _, _, d in G.edges(data=True)
        for r in d.get("relations", [])
    ))

    # Probabilities
    exp_eps = np.exp(epsilon)
    p_keep  = exp_eps / (exp_eps + 1)   # keep existing edge
    q_add   = 1.0 / (exp_eps + 1)       # add non-existing edge

    nodes = list(G.nodes())
    n     = len(nodes)

    kept    = 0
    removed = 0
    added   = 0

    # Process existing edges
    print("      Processing existing edges ...")
    for u, v, data in G.edges(data=True):
        if random.random() < p_keep:
            G_prime.add_edge(u, v, relations=data.get("relations", ["unknown"])[:])
            kept += 1
        else:
            removed += 1

    # Process non-edges (sample for large graphs)
    print("      Processing non-edges ...")
    n_edges     = G.number_of_edges()
    n_possible  = n * (n - 1) // 2
    n_nonedges  = n_possible - n_edges

    # For large graphs: sample non-edges instead of iterating all
    if sample_nonedges < 1.0 or n_nonedges > 5_000_000:
        # Sample approach: estimate how many non-edges to add
        # Expected additions = q_add * n_nonedges
        n_to_sample = min(int(n_nonedges * sample_nonedges), 10_000_000)
        n_expected_add = int(q_add * n_to_sample)
        print(f"      Sampling {n_to_sample:,} non-edges "
              f"(expected additions: {n_expected_add:,}) ...")

        attempts = 0
        max_attempts = n_expected_add * 20
        while added < n_expected_add and attempts < max_attempts:
            attempts += 1
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u != v and not G.has_edge(u, v) and not G_prime.has_edge(u, v):
                rel = random.choice(all_relations)
                G_prime.add_edge(u, v, relations=[rel])
                added += 1
    else:
        # Small graph: iterate all non-edges
        print(f"      Iterating all {n_nonedges:,} non-edges ...")
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if not G.has_edge(u, v):
                    if random.random() < q_add:
                        rel = random.choice(all_relations)
                        G_prime.add_edge(u, v, relations=[rel])
                        added += 1

    print(f"      Edges kept   : {kept:,}")
    print(f"      Edges removed: {removed:,}")
    print(f"      Edges added  : {added:,}")
    print(f"      Final edges  : {G_prime.number_of_edges():,}")

    # Privacy guarantee summary
    print(f"\n      📊 Privacy Guarantee:")
    print(f"      ε = {epsilon} → any attacker's advantage is bounded by e^ε = {exp_eps:.3f}")
    print(f"      Interpretation: P(correct inference on G) ≤ {exp_eps:.3f} × P(correct on G')")

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
    G_prime = randomized_response(
        G,
        epsilon=args.epsilon,
        seed=args.seed,
        sample_nonedges=args.sample_nonedges
    )
    metrics = measure_utility(G, G_prime)
    save_graph(G_prime, args.output)

    print(f"\n{'='*55}")
    print(f"  RANDOMIZED RESPONSE DEFENSE DONE (ε={args.epsilon})")
    print(f"  edge_overlap={metrics['edge_overlap']:.4f}  "
          f"kl_div={metrics['degree_kl_div']:.4f}")
    print(f"{'='*55}")
    print(f"\nNext: run your attacks on {args.output}")


if __name__ == "__main__":
    main()