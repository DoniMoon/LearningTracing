# infer_prior_into_embeddings.py
import argparse
import json
from typing import Any, Dict, List

import numpy as np
import joblib


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", type=str, default="output_with_embedding_octen.json")
    ap.add_argument("--model_path", type=str, default="ridge_embed2acc.joblib")
    ap.add_argument("--out_json", type=str, default="output_with_embedding_octen_with_prior.json")
    ap.add_argument("--embedding_key", type=str, default=None)  # override if you want
    ap.add_argument("--prior_key", type=str, default="prior")
    ap.add_argument("--batch_size", type=int, default=4096)
    args = ap.parse_args()

    data = load_json(args.in_json)

    bundle = joblib.load(args.model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    embedding_key = args.embedding_key
    if embedding_key is None and isinstance(bundle, dict):
        embedding_key = bundle.get("embedding_key", "Embeddings_octen")
    if embedding_key is None:
        embedding_key = "Embeddings_octen"

    total_kc = 0
    missing_embed = 0
    bad_embed = 0

    # Gather all embeddings with pointers so we can batch inference
    flat_embs: List[np.ndarray] = []
    flat_ptrs: List[tuple[str, int]] = []  # (skill_id, kc_i)

    for skill_id, rec in data.items():
        embs = rec.get(embedding_key, None)
        if not isinstance(embs, list) or len(embs) == 0:
            missing_embed += 1
            # still write empty prior if you want; here: skip
            continue

        # initialize prior list (same length as embeddings)
        prior_list = [None] * len(embs)
        rec[args.prior_key] = prior_list

        for i, e in enumerate(embs):
            total_kc += 1
            try:
                v = np.asarray(e, dtype=np.float32)
                if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
                    bad_embed += 1
                    continue
                flat_embs.append(v)
                flat_ptrs.append((skill_id, i))
            except Exception:
                bad_embed += 1
                continue

    if len(flat_embs) == 0:
        raise RuntimeError("No valid embeddings found to run inference on.")

    X = np.stack(flat_embs, axis=0)

    # Batch predict and write back
    bs = max(1, int(args.batch_size))
    for s in range(0, X.shape[0], bs):
        chunk = X[s : s + bs]
        pred = model.predict(chunk)
        pred = np.clip(pred, 0.0, 1.0)

        for j, p in enumerate(pred):
            skill_id, kc_i = flat_ptrs[s + j]
            data[skill_id][args.prior_key][kc_i] = float(p)

    # Optional sanity: fill any remaining None with NaN
    filled_none = 0
    for _, rec in data.items():
        if args.prior_key not in rec:
            continue
        pr = rec[args.prior_key]
        for i, v in enumerate(pr):
            if v is None:
                pr[i] = float("nan")
                filled_none += 1

    print(f"[INFO] embedding_key={embedding_key}, prior_key={args.prior_key}")
    print(f"[INFO] skills={len(data)} total_kc_seen={total_kc}")
    print(f"[INFO] valid_kc_inferred={X.shape[0]}")
    print(f"[INFO] missing_embed_skills={missing_embed} bad_embed_kc={bad_embed} filled_none={filled_none}")

    save_json(data, args.out_json)
    print(f"[INFO] Saved: {args.out_json}")


if __name__ == "__main__":
    main()
