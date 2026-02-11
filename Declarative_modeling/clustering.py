#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clustering.py

Input:
  - output_with_embedding_octen.json
    schema: {skill_id: {"KCs": [str], "Embeddings_octen": [[float]]}}  # embedding per KC

Steps:
  1) Build global KC pool (unique by KC string), keep one embedding per KC string (first occurrence).
  2) Normalize embeddings, union any pair with cosine >= sim_threshold (default 0.75)
     - Uses sklearn NearestNeighbors radius search (cosine distance <= 1 - threshold)
  3) Collapse to union-components (canonical KC ids)
  4) KMeans on component centroids to create K clusters (K in {25,50,75,100} by default)
  5) Save mapping + per-skill clustered labels

Outputs:
  - out_dir/clustering_k{K}.json for each K:
    {
      "meta": {...},
      "kc_string_to_canon": {kc_str: canon_id},
      "canon_to_cluster": {canon_id: cluster_id},
      "skill_to_clusters": {skill_id: [cluster_id, ...]}  # same length as original KCs list
    }
"""

import argparse
import json
import os
from collections import defaultdict
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
    """
    Returns:
      kc_list: unique KC strings
      emb: (N, D)
      kc_to_idx: KC string -> idx
      skill_to_kcs: skill_id -> original KC string list (order preserved)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kc_to_idx: Dict[str, int] = {}
    kc_list: List[str] = []
    emb_list: List[List[float]] = []
    skill_to_kcs: Dict[str, List[str]] = {}

    missing_emb = 0
    for skill_id, obj in data.items():
        kcs = obj.get("KCs", [])
        embs = obj.get(emb_key, None)

        skill_to_kcs[str(skill_id)] = [str(k) for k in kcs]

        if embs is None:
            missing_emb += 1
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

    if missing_emb > 0:
        print(f"[WARN] {missing_emb} skills missing embeddings key={emb_key} (ignored for pooling).")

    emb = np.asarray(emb_list, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape={emb.shape}")
    return kc_list, emb, kc_to_idx, skill_to_kcs


def build_union_by_radius(emb_norm: np.ndarray, sim_threshold: float, leaf_size: int = 40, n_jobs: int = -1) -> DSU:
    """
    Union any pairs with cosine similarity >= sim_threshold using radius search on cosine distance.
    cosine_distance = 1 - cosine_similarity
    radius = 1 - sim_threshold
    """
    from sklearn.neighbors import NearestNeighbors

    n = emb_norm.shape[0]
    dsu = DSU(n)

    radius = 1.0 - float(sim_threshold)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=n_jobs)
    nn.fit(emb_norm)

    # radius_neighbors returns (distances, indices) per point
    # For large N, do it in batches to avoid huge memory spikes.
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


def collapse_components(dsu: DSU, emb_norm: np.ndarray) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    """
    Returns:
      canon_id_per_kc: (N,) canonical component id (0..C-1)
      root_to_canon: root -> canon_id
      canon_centroids: (C, D)
    """
    n, d = emb_norm.shape
    roots = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)
    uniq_roots, inv = np.unique(roots, return_inverse=True)  # inv is canon-ish but not compact by root order

    # Make canon ids compact by uniq_roots order
    root_to_canon = {int(r): int(ci) for ci, r in enumerate(uniq_roots)}
    canon_id_per_kc = np.array([root_to_canon[int(r)] for r in roots], dtype=np.int64)

    # centroid per canon
    C = len(uniq_roots)
    canon_sum = np.zeros((C, d), dtype=np.float32)
    canon_cnt = np.zeros((C,), dtype=np.int64)
    for i in range(n):
        c = canon_id_per_kc[i]
        canon_sum[c] += emb_norm[i]
        canon_cnt[c] += 1
    canon_centroids = canon_sum / np.maximum(canon_cnt[:, None], 1)

    canon_centroids = l2_normalize(canon_centroids)
    return canon_id_per_kc, root_to_canon, canon_centroids


def run_kmeans(canon_centroids: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    # embeddings are unit norm; euclidean kmeans ~= cosine clustering
    labels = km.fit_predict(canon_centroids)
    return labels.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=str, required=True)
    ap.add_argument("--emb_key", type=str, default="Embeddings_octen")
    ap.add_argument("--sim_threshold", type=float, default=0.75)
    ap.add_argument("--k_list", type=str, default="25,50,75,100")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="./kc_labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]

    kc_list, emb, kc_to_idx, skill_to_kcs = load_kc_pool(args.input_json, args.emb_key)
    print(f"[INFO] Unique KC strings: {len(kc_list)}, embedding dim: {emb.shape[1]}")

    emb_norm = l2_normalize(emb)

    print(f"[INFO] Union by cosine >= {args.sim_threshold}")
    dsu = build_union_by_radius(emb_norm, args.sim_threshold)

    canon_id_per_kc, root_to_canon, canon_centroids = collapse_components(dsu, emb_norm)
    num_canon = canon_centroids.shape[0]
    print(f"[INFO] Canonical KC components: {num_canon}")

    # map KC string -> canon_id
    kc_string_to_canon = {kc: int(canon_id_per_kc[idx]) for kc, idx in kc_to_idx.items()}

    for k in k_list:
        if k > num_canon:
            print(f"[WARN] Skip K={k} because num_canon={num_canon} < K")
            continue

        print(f"[INFO] KMeans clustering K={k}")
        canon_cluster = run_kmeans(canon_centroids, k=k, seed=args.seed)
        canon_to_cluster = {str(ci): int(canon_cluster[ci]) for ci in range(num_canon)}

        # per skill_id list of cluster ids (preserve original KC list length)
        skill_to_clusters: Dict[str, List[int]] = {}
        missing = 0
        for skill_id, kcs in skill_to_kcs.items():
            out = []
            for kc in kcs:
                if kc not in kc_string_to_canon:
                    missing += 1
                    out.append(-1)
                    continue
                canon = kc_string_to_canon[kc]
                out.append(int(canon_cluster[canon]))
            skill_to_clusters[skill_id] = out

        if missing > 0:
            print(f"[WARN] Missing KC strings in pool: {missing} (labeled as -1).")

        out_path = os.path.join(args.out_dir, f"clustering_k{k}.json")
        payload = {
            "meta": {
                "input_json": args.input_json,
                "emb_key": args.emb_key,
                "sim_threshold": args.sim_threshold,
                "k": k,
                "seed": args.seed,
                "num_unique_kc": len(kc_list),
                "num_canon": num_canon,
            },
            "kc_string_to_canon": {k: int(v) for k, v in kc_string_to_canon.items()},
            "canon_to_cluster": canon_to_cluster,
            "skill_to_clusters": skill_to_clusters,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
