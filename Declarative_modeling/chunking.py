#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chunking.py

Input:
  - output_with_embedding_octen.json
    {skill_id: {"KCs":[str], "Embeddings_octen":[[float]]}}

Steps:
  1) Build global KC pool (unique KC string), normalize embeddings
  2) Canonicalize by cosine>=0.75 union (same as clustering)
  3) Scan each skill's KC list (after canon) to count:
       C(A): occurrences of A
       C(AB): occurrences of consecutive pair A->B
  4) For each threshold t in T:
       Merge rule: if C(AB)/C(A) > t, then map A into B (directional merge)
       Merge cap: each B can absorb at most n merges (in-degree cap)
     Implementation uses DSU-like structure with forced representative = B
     while preventing cycles.

Outputs:
  - out_dir/chunking_t{t}_n{n}.json
    {
      "meta": {...},
      "kc_string_to_canon": {kc_str: canon_id},
      "canon_redirect": {canon_id: final_id},     # after chunk merges (0..C-1)
      "skill_to_chunked": {skill_id: [final_id,...]}  # aligned with original KC list
    }
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


class DSU:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.int8)

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def load_kc_pool(json_path: str, emb_key: str) -> Tuple[List[str], np.ndarray, Dict[str, int], Dict[str, List[str]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kc_to_idx: Dict[str, int] = {}
    kc_list: List[str] = []
    emb_list: List[List[float]] = []
    skill_to_kcs: Dict[str, List[str]] = {}

    for skill_id, obj in data.items():
        kcs = obj.get("KCs", [])
        embs = obj.get(emb_key, None)
        skill_to_kcs[str(skill_id)] = [str(k) for k in kcs]

        if embs is None:
            continue
        if len(kcs) != len(embs):
            raise ValueError(f"Length mismatch at skill_id={skill_id}: len(KCs)={len(kcs)} vs len({emb_key})={len(embs)}")

        for kc, e in zip(kcs, embs):
            kc = str(kc)
            if kc in kc_to_idx:
                continue
            kc_to_idx[kc] = len(kc_list)
            kc_list.append(kc)
            emb_list.append(e)

    emb = np.asarray(emb_list, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape={emb.shape}")
    return kc_list, emb, kc_to_idx, skill_to_kcs


def build_union_by_radius(emb_norm: np.ndarray, sim_threshold: float, n_jobs: int = -1) -> DSU:
    from sklearn.neighbors import NearestNeighbors

    n = emb_norm.shape[0]
    dsu = DSU(n)
    radius = 1.0 - float(sim_threshold)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=n_jobs)
    nn.fit(emb_norm)

    batch = 4096
    for start in range(0, n, batch):
        end = min(n, start + batch)
        X = emb_norm[start:end]
        neigh_idx = nn.radius_neighbors(X, radius=radius, return_distance=False)
        for i_local, nbrs in enumerate(neigh_idx):
            i = start + i_local
            for j in nbrs:
                if j <= i:
                    continue
                dsu.union(i, int(j))
        if (start // batch) % 10 == 0:
            print(f"[INFO] radius-union progress: {end}/{n}")

    return dsu


def collapse_components(dsu: DSU) -> Tuple[np.ndarray, Dict[int, int]]:
    n = len(dsu.p)
    roots = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    uniq_roots = np.unique(roots)
    root_to_canon = {int(r): int(ci) for ci, r in enumerate(uniq_roots)}
    canon_id_per_kc = np.array([root_to_canon[int(r)] for r in roots], dtype=np.int64)
    return canon_id_per_kc, root_to_canon


class ForcedRepUF:
    """
    Directional-ish merging: we want to "merge A into B" meaning rep(A)=rep(B)=B's rep.
    We'll keep a parent array where find() returns representative.
    union_into(a, b) forces rep(a) to become rep(b), unless it creates a cycle.
    """
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union_into(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        # force ra -> rb
        self.p[ra] = rb
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=str, required=True)
    ap.add_argument("--emb_key", type=str, default="Embeddings_octen")
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--t_list", type=str, default="0.6,0.7,0.8,0.9")
    ap.add_argument("--n_list", type=str, default="5,10,20")
    ap.add_argument("--out_dir", type=str, default="./kc_labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_list = [float(x.strip()) for x in args.t_list.split(",") if x.strip()]
    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]

    kc_list, emb, kc_to_idx, skill_to_kcs = load_kc_pool(args.input_json, args.emb_key)
    print(f"[INFO] Unique KC strings: {len(kc_list)}, embedding dim: {emb.shape[1]}")

    emb_norm = l2_normalize(emb)
    print(f"[INFO] Union by cosine >= {args.sim_threshold}")
    dsu = build_union_by_radius(emb_norm, args.sim_threshold)
    canon_id_per_kc, root_to_canon = collapse_components(dsu)
    num_canon = int(np.max(canon_id_per_kc)) + 1
    print(f"[INFO] Canonical KC components: {num_canon}")

    kc_string_to_canon = {kc: int(canon_id_per_kc[idx]) for kc, idx in kc_to_idx.items()}

    # Build per skill canon sequences
    skill_to_canon_seq: Dict[str, List[int]] = {}
    for skill_id, kcs in skill_to_kcs.items():
        seq = []
        for kc in kcs:
            if kc not in kc_string_to_canon:
                seq.append(-1)
            else:
                seq.append(kc_string_to_canon[kc])
        skill_to_canon_seq[skill_id] = seq

    # Count C(A), C(AB) over all skills, ignoring -1
    C_A = Counter()
    C_AB = Counter()
    for skill_id, seq in skill_to_canon_seq.items():
        seq = [x for x in seq if x != -1]
        for a in seq:
            C_A[a] += 1
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a == b:
                continue
            C_AB[(a, b)] += 1

    # Precompute candidate edges with ratio
    edges = []
    for (a, b), cab in C_AB.items():
        ca = C_A[a]
        if ca <= 0:
            continue
        ratio = cab / ca
        edges.append((ratio, a, b, cab, ca))
    edges.sort(reverse=True, key=lambda x: x[0])
    print(f"[INFO] Candidate directed edges: {len(edges)}")

    for t in t_list:
        for ncap in n_list:
            uf = ForcedRepUF(num_canon)
            absorbed_count = np.zeros(num_canon, dtype=np.int32)  # how many nodes merged into this rep

            merges = 0
            for ratio, a, b, cab, ca in edges:
                if ratio <= t:
                    break

                ra = uf.find(a)
                rb = uf.find(b)
                if ra == rb:
                    continue

                # cap: rb can absorb at most ncap merges
                if absorbed_count[rb] >= ncap:
                    continue

                # prevent 2-cycle-ish behavior: if rb eventually points to ra (cycle)
                # since our uf only points upward (compression), a cycle is unlikely unless forced,
                # but we still guard by checking root chain.
                x = rb
                cycle = False
                while uf.p[x] != x:
                    x = uf.p[x]
                    if x == ra:
                        cycle = True
                        break
                if cycle:
                    continue

                # merge ra into rb
                uf.union_into(ra, rb)
                absorbed_count[rb] += 1
                merges += 1

            # finalize redirect mapping
            canon_redirect = {str(i): int(uf.find(i)) for i in range(num_canon)}

            # per-skill final ids aligned with original KCs list
            skill_to_chunked: Dict[str, List[int]] = {}
            for skill_id, seq in skill_to_canon_seq.items():
                out = []
                for x in seq:
                    if x == -1:
                        out.append(-1)
                    else:
                        out.append(int(uf.find(x)))
                skill_to_chunked[skill_id] = out

            out_path = os.path.join(args.out_dir, f"chunking_t{t}_n{ncap}.json")
            payload = {
                "meta": {
                    "input_json": args.input_json,
                    "emb_key": args.emb_key,
                    "sim_threshold": args.sim_threshold,
                    "t": t,
                    "ncap": ncap,
                    "num_unique_kc": len(kc_list),
                    "num_canon": num_canon,
                    "num_merges": merges,
                },
                "kc_string_to_canon": {k: int(v) for k, v in kc_string_to_canon.items()},
                "canon_redirect": canon_redirect,
                "skill_to_chunked": skill_to_chunked,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            print(f"[OK] wrote {out_path} (merges={merges})")


if __name__ == "__main__":
    main()
