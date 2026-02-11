#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_learning_curves.py

Goal:
  For each labeling method output (clustering_kK.json or chunking_t*_n*.json),
  compute the ratio of KCs whose power-law fit achieves R^2 >= r2_threshold (default 0.3),
  among "valid" KCs defined as:
    - at least min_users users (default 30)
    - each of those users has at least min_attempts_per_user attempts on that KC (default 30)

Data:
  - preprocessed_data.csv (TSV): question_id, user_id, item_id, timestamp, correct, skill_id
  - label json:
      clustering: has skill_to_clusters
      chunking:   has skill_to_chunked

Learning curve & power law:
  - For each KC, we build per-user attempt sequence (ordered by timestamp)
  - For attempt i=1..min_attempts_per_user:
      acc_i = mean(correct at user's i-th attempt) over users who have >= i attempts
      err_i = 1 - acc_i
  - Fit power-law on err: err_i = a * i^b   (log-log linear regression)
      log(err_i) = log(a) + b*log(i)
    We compute R^2 in log space (common for power-law evaluation).
  - KC is counted as "fits" if R^2 >= r2_threshold.

Outputs:
  - Prints a LaTeX-ish table rows + also dumps a JSON summary if --out_json is provided.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def read_tsv(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def load_label_json(path: str) -> Tuple[str, Dict[str, List[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "skill_to_clusters" in obj:
        return "clustering", obj["skill_to_clusters"]
    if "skill_to_chunked" in obj:
        return "chunking", obj["skill_to_chunked"]
    raise ValueError(f"Unrecognized label json schema: {path}")


def powerlaw_r2_loglog(err: np.ndarray, x: np.ndarray, eps: float = 1e-9) -> float:
    """
    Fit log(err) = c + b*log(x) via least squares, return R^2 in log space.
    """
    err = np.asarray(err, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    # Filter invalid
    mask = (err > 0) & np.isfinite(err) & (x > 0) & np.isfinite(x)
    err = err[mask]
    x = x[mask]
    if len(err) < 3:
        return float("nan")

    y = np.log(np.maximum(err, eps))
    lx = np.log(x)

    A = np.vstack([np.ones_like(lx), lx]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def evaluate_labeling(
    data_rows: List[dict],
    skill_to_labels: Dict[str, List[int]],
    min_attempts_per_user: int = 30,
    min_users: int = 30,
    r2_threshold: float = 0.3,
) -> Tuple[float, int, int]:
    """
    Returns:
      ratio_fit, num_fit, num_valid
    """
    # Build KC -> user -> list of (timestamp, correct)
    kc_user_events: Dict[int, Dict[str, List[Tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))

    missing_skill = 0
    missing_label = 0

    for r in data_rows:
        skill_id = str(r["skill_id"])
        user_id = str(r["user_id"])
        ts = float(r["timestamp"])
        correct = int(float(r["correct"]))

        if skill_id not in skill_to_labels:
            missing_skill += 1
            continue
        labels = skill_to_labels[skill_id]
        if not labels:
            missing_label += 1
            continue

        # One interaction can map to multiple KCs (labels list).
        # We treat this as exposure to each KC (multi-label).
        for kc in labels:
            if kc is None:
                continue
            kc = int(kc)
            if kc < 0:
                continue
            kc_user_events[kc][user_id].append((ts, correct))

    if missing_skill > 0:
        print(f"[WARN] {missing_skill} rows have skill_id not in label mapping (ignored).")
    if missing_label > 0:
        print(f"[WARN] {missing_label} rows have empty label list (ignored).")

    num_valid = 0
    num_fit = 0

    x = np.arange(1, min_attempts_per_user + 1, dtype=np.float64)

    for kc, user_events in kc_user_events.items():
        # Build per-user ordered correctness sequence
        user_seqs = []
        for uid, evs in user_events.items():
            evs.sort(key=lambda t: t[0])
            seq = [c for _, c in evs]
            if len(seq) >= min_attempts_per_user:
                user_seqs.append(seq[:min_attempts_per_user])

        if len(user_seqs) < min_users:
            continue

        num_valid += 1
        M = np.asarray(user_seqs, dtype=np.float64)  # (U, 30)
        acc = np.mean(M, axis=0)
        err = 1.0 - acc

        r2 = powerlaw_r2_loglog(err=err, x=x)
        if np.isfinite(r2) and r2 >= r2_threshold:
            num_fit += 1

    ratio = (num_fit / num_valid) if num_valid > 0 else float("nan")
    return ratio, num_fit, num_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_tsv", type=str, required=True)
    ap.add_argument("--label_jsons", type=str, nargs="+", required=True)
    ap.add_argument("--min_attempts_per_user", type=int, default=30)
    ap.add_argument("--min_users", type=int, default=30)
    ap.add_argument("--r2_threshold", type=float, default=0.3)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    rows = read_tsv(args.data_tsv)
    print(f"[INFO] Loaded TSV rows: {len(rows)}")

    results = []

    for label_path in args.label_jsons:
        kind, skill_to_labels = load_label_json(label_path)
        ratio, num_fit, num_valid = evaluate_labeling(
            data_rows=rows,
            skill_to_labels=skill_to_labels,
            min_attempts_per_user=args.min_attempts_per_user,
            min_users=args.min_users,
            r2_threshold=args.r2_threshold,
        )
        rec = {
            "label_path": label_path,
            "kind": kind,
            "ratio_fit": ratio,
            "num_fit": num_fit,
            "num_valid": num_valid,
        }
        results.append(rec)
        print(f"[RESULT] {os.path.basename(label_path)}  ratio={ratio:.4f}  fit={num_fit}  valid={num_valid}")

    # Print helper tables
    # clustering table by K
    clustering = []
    chunking = []
    for r in results:
        base = os.path.basename(r["label_path"])
        if r["kind"] == "clustering":
            # expects clustering_k{K}.json
            k = None
            for token in base.split("_"):
                if token.startswith("k") and token[1:].replace(".json", "").isdigit():
                    k = int(token[1:].replace(".json", ""))
            if k is None:
                # fallback parse
                try:
                    k = int(base.split("clustering_k")[1].split(".json")[0])
                except Exception:
                    k = -1
            clustering.append((k, r["ratio_fit"]))
        else:
            # expects chunking_t{t}_n{n}.json
            t = None
            ncap = None
            try:
                mid = base.replace(".json", "")
                # chunking_t0.7_n10
                part = mid.split("chunking_t")[1]
                t_str, n_str = part.split("_n")
                t = float(t_str)
                ncap = int(n_str)
            except Exception:
                t, ncap = float("nan"), -1
            chunking.append((t, ncap, r["ratio_fit"]))

    if clustering:
        clustering.sort(key=lambda x: x[0])
        print("\n[CLUSTERING TABLE] (R^2>=threshold ratio among valid KCs)")
        header = "K & " + " & ".join(str(k) for k, _ in clustering) + r" \\"
        row = "ratio & " + " & ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for _, v in clustering) + r" \\"
        print(header)
        print(row)

    if chunking:
        chunking.sort(key=lambda x: (x[0], x[1]))
        print("\n[CHUNKING TABLE] rows=t, cols=ncap (values=ratio)")
        # Build grid
        ts = sorted({t for t, _, _ in chunking})
        ns = sorted({n for _, n, _ in chunking})
        grid = {(t, n): v for t, n, v in chunking}
        print("t/ncap & " + " & ".join(str(n) for n in ns) + r" \\")
        for t in ts:
            vals = [grid.get((t, n), float("nan")) for n in ns]
            line = f"{t:.1f} & " + " & ".join(f"{v:.3f}" if np.isfinite(v) else "nan" for v in vals) + r" \\"
            print(line)

    if args.out_json is not None:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"meta": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
