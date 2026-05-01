#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import Counter

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_rel_name(rel: str) -> str:
    rel = rel.strip().strip("/")
    rel = rel.replace("/", "__").replace("\\", "__")
    return rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global_path", required=True,
                    help="Path to global KG file (TSV: h\\tr\\tt)")
    ap.add_argument("--relation", action="append", required=True,
                    help="Sensitive relation (repeat --relation multiple times)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default utf-8)")
    args = ap.parse_args()

    global_path = Path(args.global_path)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    sens_rels = sorted(set([r.strip() for r in args.relation if r and r.strip()]))
    if not sens_rels:
        raise SystemExit("No --relation provided.")

    public_out = outdir / "global_kg_public_wo_sensitive.tsv"
    sens_dir = outdir / "sensitive"
    ensure_dir(sens_dir)

    rel2file = {r: sens_dir / f"{safe_rel_name(r)}.tsv" for r in sens_rels}

    print(f"[+] Input file: {global_path}")
    print(f"[+] Sensitive relations ({len(sens_rels)}): {sens_rels}")
    print(f"[+] Public output (wo ALL sensitive): {public_out}")
    print(f"[+] Sensitive outputs dir: {sens_dir}")

    sens_handles = {}
    try:
        for r in sens_rels:
            fp = rel2file[r]
            print(f"    -> writing sensitive triples for {r} to {fp}")
            sens_handles[r] = fp.open("w", encoding=args.encoding, errors="ignore")

        n_total = 0
        n_public = 0
        n_sensitive = Counter()
        rel_counts = Counter()

        with global_path.open("r", encoding=args.encoding, errors="ignore") as fin, \
             public_out.open("w", encoding=args.encoding, errors="ignore") as fpub:

            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    parts = line.split()
                    if len(parts) < 3:
                        continue

                h, r, t = parts[0], parts[1], parts[2]
                n_total += 1
                rel_counts[r] += 1

                if r in rel2file:
                    sens_handles[r].write(f"{h}\t{r}\t{t}\n")
                    n_sensitive[r] += 1
                else:
                    fpub.write(f"{h}\t{r}\t{t}\n")
                    n_public += 1

                if n_total % 5_000_000 == 0:
                    print(f"  processed {n_total:,} triples ...")

        report_path = outdir / "split_report_multi.txt"
        with report_path.open("w", encoding=args.encoding, errors="ignore") as rep:
            rep.write(f"INPUT\t{global_path}\n")
            rep.write(f"TOTAL_TRIPLES\t{n_total}\n")
            rep.write(f"PUBLIC_TRIPLES\t{n_public}\n")
            rep.write(f"SENSITIVE_RELATIONS\t{len(sens_rels)}\n")
            rep.write("\n# Sensitive counts per relation\n")
            for r in sens_rels:
                rep.write(f"{r}\t{n_sensitive[r]}\tfile={rel2file[r].name}\n")
            rep.write("\n# Top relations in original graph (by count)\n")
            for r, c in rel_counts.most_common(50):
                rep.write(f"{c}\t{r}\n")

        print("\n[+] Done.")
        print(f"[+] Report saved to: {report_path}")
        print(f"[+] Public graph saved to: {public_out}")
        print(f"[+] Sensitive files in: {sens_dir}")

        print("\n[+] Sensitive file names (safe):")
        for r in sens_rels:
            print(f"  {r}  ->  {rel2file[r].name}")

    finally:
        for f in sens_handles.values():
            try:
                f.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
