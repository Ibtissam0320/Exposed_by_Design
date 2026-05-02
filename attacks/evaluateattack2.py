#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack2 Evaluation (Open-world-ish pool) — Multi-tail Ground Truth
Compute: Coverage(mean), Hits@K, MRR

This script evaluates Attack2 pairwise scoring outputs against a sensitive (ground-truth) file
that can have MULTIPLE tails per head.

It supports two common score TSV formats:

FORMAT A (recommended / "long" format):
    head_id    tail_id    score    [rank]
(one row per scored tail per head; rank optional. If rank absent, it is computed by sorting score desc per head)

FORMAT B ("summary" format from your earlier single-tail script):
    head_id    true_tail_id    pred_tail_id    rank_true    ...
In this case, we can compute Hits@K/MRR ONLY for the single true_tail_id using rank_true.
(For multi-tail GT, FORMAT B is not sufficient.)

Ground truth file (sensitive):
    head_id    rel_id    tail_id
(no header, tab-separated)

Coverage(mean) here means:
    fraction of heads for which at least one GT tail appears in the scored candidate set (pool) for that head.
If you scored ALL tails in the pool (topk=0), coverage should be high.
If you only saved topk (e.g., 200), coverage is "coverage within saved candidates".

Outputs:
  - prints metrics
  - optionally writes a JSON metrics file

Examples:
  # Long format (topk saved or full pool)
  python3 eval_attack2_mrr_hits_coverage.py \
    --sens-path /path/to/sensitive/religion_ids.tsv \
    --scores-path /path/to/scores/62_attack2_pair_scores.tsv \
    --relation-filter 62 \
    --ks 1,3,5,10,50,100,200 \
    --out-metrics /path/to/metrics/eval_metrics.json

  # If your scores file is summary format with rank_true:
  python3 eval_attack2_mrr_hits_coverage.py \
    --sens-path /path/to/sensitive/religion_ids.tsv \
    --scores-path /path/to/scores/religion_attack2_pair_scores.tsv \
    --relation-filter 62
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def as_id(x):
    if pd.isna(x):
        return None
    return str(x).strip()


def load_ground_truth_multitail(sens_path: Path, relation_filter: str | None):
    df = pd.read_csv(
        sens_path,
        sep="\t",
        header=None,
        names=["head_id", "rel_id", "tail_id"],
        dtype=str,
        low_memory=False,
    ).dropna()

    df["head_id"] = df["head_id"].map(as_id)
    df["rel_id"] = df["rel_id"].map(as_id)
    df["tail_id"] = df["tail_id"].map(as_id)
    df = df.dropna()

    if relation_filter is not None and relation_filter != "":
        df = df[df["rel_id"] == str(relation_filter)]

    if df.empty:
        raise RuntimeError("Ground truth is empty after relation_filter. Check rel_id values / filter.")

    head2tails = defaultdict(set)
    for h, t in df[["head_id", "tail_id"]].itertuples(index=False, name=None):
        head2tails[h].add(t)

    return head2tails


def parse_ks(s: str):
    ks = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x.isdigit():
            ks.append(int(x))
    ks = sorted(set(k for k in ks if k > 0))
    if not ks:
        ks = [1, 3, 5, 10]
    return ks


def eval_from_long_scores(df_scores: pd.DataFrame, head2tails: dict, ks: list[int]):
    """
    df_scores must contain: head_id, tail_id, score (rank optional)
    """
    needed = {"head_id", "tail_id", "score"}
    if not needed.issubset(set(df_scores.columns)):
        raise RuntimeError(f"Long format requires columns {sorted(needed)}. Found: {list(df_scores.columns)}")

    df = df_scores.copy()
    df["head_id"] = df["head_id"].map(as_id)
    df["tail_id"] = df["tail_id"].map(as_id)
    df["score"] = df["score"].astype(float)
    df = df.dropna(subset=["head_id", "tail_id", "score"])

    # If rank not present, compute rank per head by sorting score desc (ties broken by tail_id)
    if "rank" not in df.columns:
        df = df.sort_values(["head_id", "score", "tail_id"], ascending=[True, False, True])
        df["rank"] = df.groupby("head_id").cumcount() + 1
    else:
        df["rank"] = df["rank"].astype(int)

    # Evaluate only heads that exist in ground truth
    heads_scored = df["head_id"].unique().tolist()
    heads_eval = [h for h in heads_scored if h in head2tails]

    if len(heads_eval) == 0:
        raise RuntimeError("No overlap between scored heads and ground truth heads.")

    # Build per-head arrays
    hits = {k: [] for k in ks}
    rr_list = []
    coverage_list = []

    # To speed up: group once
    g = df[df["head_id"].isin(heads_eval)].groupby("head_id", sort=False)

    for h, block in g:
        gt = head2tails.get(h, set())
        if not gt:
            continue

        # candidates present in score file for this head
        cand_tails = set(block["tail_id"].tolist())
        covered = 1.0 if len(gt.intersection(cand_tails)) > 0 else 0.0
        coverage_list.append(covered)

        # ranks of GT tails that appear in candidates
        gt_block = block[block["tail_id"].isin(gt)]
        if len(gt_block) == 0:
            # not found in saved candidates => MRR 0, hits 0
            rr_list.append(0.0)
            for k in ks:
                hits[k].append(0.0)
            continue

        best_rank = int(gt_block["rank"].min())
        rr_list.append(1.0 / best_rank)

        for k in ks:
            hits[k].append(1.0 if best_rank <= k else 0.0)

    metrics = {
        "n_heads_eval": int(len(rr_list)),
        "coverage_mean": float(np.mean(coverage_list)) if coverage_list else 0.0,
        "mrr": float(np.mean(rr_list)) if rr_list else 0.0,
        "hits": {f"hits@{k}": float(np.mean(hits[k])) if hits[k] else 0.0 for k in ks},
    }
    return metrics


def eval_from_summary_scores(df_scores: pd.DataFrame, head2tails: dict, ks: list[int]):
    """
    Summary format: needs head_id + rank_true (single GT tail only).
    For multi-tail GT, this is not enough, but we still compute metrics for the provided true_tail_id/rank_true.
    """
    if "head_id" not in df_scores.columns or "rank_true" not in df_scores.columns:
        raise RuntimeError("Summary format requires columns head_id and rank_true.")

    df = df_scores.copy()
    df["head_id"] = df["head_id"].map(as_id)
    df["rank_true"] = df["rank_true"].astype(int)
    df = df.dropna(subset=["head_id", "rank_true"])

    # Evaluate only heads that exist in ground truth
    df = df[df["head_id"].isin(set(head2tails.keys()))]
    if df.empty:
        raise RuntimeError("No overlap between summary-scored heads and ground truth heads.")

    ranks = df["rank_true"].to_numpy(dtype=np.int64)
    rr = (1.0 / ranks).astype(float)

    out = {
        "n_heads_eval": int(len(ranks)),
        # in summary format we don't know the candidate pool membership per head, so coverage can't be computed correctly
        "coverage_mean": None,
        "mrr": float(np.mean(rr)) if len(rr) else 0.0,
        "hits": {f"hits@{k}": float(np.mean(ranks <= k)) if len(ranks) else 0.0 for k in ks},
        "note": "coverage_mean is None because summary format does not provide candidate tails per head.",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens-path", required=True, type=str, help="Sensitive ground-truth TSV: head rel tail")
    ap.add_argument("--scores-path", required=True, type=str, help="Attack2 scores TSV (long or summary format)")
    ap.add_argument("--relation-filter", type=str, default="", help="Filter sensitive by rel_id (e.g., '62')")
    ap.add_argument("--ks", type=str, default="1,3,5,10,50,100,200", help="Comma separated K values")
    ap.add_argument("--out-metrics", type=str, default="", help="Optional path to write metrics JSON")
    args = ap.parse_args()

    sens_path = Path(args.sens_path)
    scores_path = Path(args.scores_path)
    rel_filter = args.relation_filter.strip() or None
    ks = parse_ks(args.ks)

    head2tails = load_ground_truth_multitail(sens_path, rel_filter)

    df_scores = pd.read_csv(scores_path, sep="\t", dtype=str, low_memory=False).dropna(how="all")
    df_scores.columns = [c.strip() for c in df_scores.columns]

    # Detect format
    cols = set(df_scores.columns)
    if {"head_id", "tail_id", "score"}.issubset(cols):
        metrics = eval_from_long_scores(df_scores, head2tails, ks)
        metrics["format"] = "long(head_id,tail_id,score[,rank])"
    elif {"head_id", "rank_true"}.issubset(cols):
        metrics = eval_from_summary_scores(df_scores, head2tails, ks)
        metrics["format"] = "summary(head_id,rank_true,...)"
    else:
        raise RuntimeError(
            "Unknown score file format.\n"
            "Expected either long format columns: head_id, tail_id, score\n"
            "or summary format columns: head_id, rank_true\n"
            f"Found columns: {list(df_scores.columns)}"
        )

    # Print metrics
    print("=== Attack2 Evaluation (MRR / Hits@K / Coverage) ===")
    print(f"Format          = {metrics.get('format')}")
    print(f"#Heads eval     = {metrics.get('n_heads_eval')}")
    cov = metrics.get("coverage_mean")
    if cov is None:
        print("Coverage(mean)  = None (not computable from summary format)")
    else:
        print(f"Coverage(mean)  = {cov:.4f}")
    print(f"MRR             = {metrics.get('mrr'):.4f}")
    for k in ks:
        hk = metrics["hits"].get(f"hits@{k}", 0.0)
        print(f"Hits@{k:<3d}        = {hk:.4f}")

    if args.out_metrics.strip():
        out_path = Path(args.out_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scores_path": str(scores_path),
            "sens_path": str(sens_path),
            "relation_filter": rel_filter,
            "ks": ks,
            "metrics": metrics,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[+] Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
