# train_embed2acc_multidata.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gold_from_kc_obj(kc_obj: Dict[str, Any]) -> Tuple[float, int]:
    solved = kc_obj.get("solved", {})
    if not isinstance(solved, dict) or len(solved) == 0:
        return np.nan, 0

    vals: List[int] = []
    for _, v in solved.items():
        if not isinstance(v, dict):
            continue
        ic = v.get("is_correct", None)
        if ic is None:
            continue
        try:
            vals.append(int(ic))
        except Exception:
            continue

    if len(vals) == 0:
        return np.nan, 0

    return float(np.mean(vals)), len(vals)


def build_dataset_from_pair(
    mcqs: Dict[str, Any],
    embed_db: Dict[str, Any],
    embedding_key: str,
    min_models: int,
    dataset_name: str,
) -> Tuple[List[np.ndarray], List[float], List[Tuple[str, str, str]], Dict[str, int]]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta: List[Tuple[str, str, str]] = []  # (dataset, qid, kc_idx)

    stats = {
        "kept": 0,
        "skipped_no_embed_qid": 0,
        "skipped_no_emb_list": 0,
        "skipped_bad_kcidx": 0,
        "skipped_no_gold": 0,
        "skipped_not_enough_models": 0,
    }

    for qid, kc_map in mcqs.items():
        if qid not in embed_db:
            stats["skipped_no_embed_qid"] += 1
            continue

        rec = embed_db[qid]
        embs = rec.get(embedding_key, None)
        if not isinstance(embs, list) or len(embs) == 0:
            stats["skipped_no_emb_list"] += 1
            continue

        if not isinstance(kc_map, dict):
            continue

        for kc_idx, kc_obj in kc_map.items():
            try:
                k = int(kc_idx)
            except Exception:
                stats["skipped_bad_kcidx"] += 1
                continue

            if k < 0 or k >= len(embs):
                stats["skipped_bad_kcidx"] += 1
                continue

            gold, n_models = extract_gold_from_kc_obj(kc_obj)
            if not np.isfinite(gold):
                stats["skipped_no_gold"] += 1
                continue
            if n_models < min_models:
                stats["skipped_not_enough_models"] += 1
                continue

            emb = np.asarray(embs[k], dtype=np.float32)
            if emb.ndim != 1 or emb.size == 0 or not np.all(np.isfinite(emb)):
                stats["skipped_bad_kcidx"] += 1
                continue

            X_list.append(emb)
            y_list.append(gold)
            meta.append((dataset_name, str(qid), str(kc_idx)))
            stats["kept"] += 1

    return X_list, y_list, meta, stats


def discover_dataset_dirs(root: Path, explicit: List[str]) -> List[Path]:
    if explicit:
        dirs = [root / d for d in explicit]
    else:
        # root 바로 아래의 디렉토리들 중에서, 두 파일이 모두 있는 곳만 사용
        dirs = [p for p in root.iterdir() if p.is_dir()]

    out: List[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        if (d / "mcqs_result.json").exists() and (d / "output_with_embedding_octen.json").exists():
            out.append(d)

    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument(
        "--datasets",
        type=str,
        default="",  # e.g. "507_statics,1148_biology,eedi_2020"
        help="Comma-separated dataset folder names under --root. If empty, auto-discover subfolders that contain both json files.",
    )
    ap.add_argument("--mcqs_name", type=str, default="mcqs_result.json")
    ap.add_argument("--embeds_name", type=str, default="output_with_embedding_octen.json")
    ap.add_argument("--embedding_key", type=str, default="Embeddings_octen")
    ap.add_argument("--min_models", type=int, default=1)  # set 20 if you want strict gold
    ap.add_argument("--test_size", type=float, default=0.125)  # 1/8
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out_model", type=str, default="ridge_embed2acc.joblib")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    explicit = [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets.strip() else []
    dataset_dirs = discover_dataset_dirs(root, explicit)

    if len(dataset_dirs) == 0:
        raise RuntimeError(
            f"No dataset dirs found under {root}. "
            f"Expected each dataset folder to contain {args.mcqs_name} and {args.embeds_name}."
        )

    print(f"[INFO] Found {len(dataset_dirs)} dataset dirs:")
    for d in dataset_dirs:
        print(f"  - {d.name}")

    X_all: List[np.ndarray] = []
    y_all: List[float] = []
    meta_all: List[Tuple[str, str, str]] = []

    # per-dataset stats
    for d in dataset_dirs:
        mcqs_path = d / args.mcqs_name
        embeds_path = d / args.embeds_name

        mcqs = load_json(mcqs_path)
        embed_db = load_json(embeds_path)

        X_list, y_list, meta, stats = build_dataset_from_pair(
            mcqs=mcqs,
            embed_db=embed_db,
            embedding_key=args.embedding_key,
            min_models=args.min_models,
            dataset_name=d.name,
        )

        X_all.extend(X_list)
        y_all.extend(y_list)
        meta_all.extend(meta)

        print(
            f"[INFO] {d.name}: kept={stats['kept']} "
            f"skip(no_embed_qid)={stats['skipped_no_embed_qid']} "
            f"skip(no_emb_list)={stats['skipped_no_emb_list']} "
            f"skip(bad_kcidx)={stats['skipped_bad_kcidx']} "
            f"skip(no_gold)={stats['skipped_no_gold']} "
            f"skip(min_models)={stats['skipped_not_enough_models']}"
        )

    if len(X_all) == 0:
        raise RuntimeError("No training samples were built across all datasets. Check schema alignment.")

    X = np.stack(X_all, axis=0)
    y = np.asarray(y_all, dtype=np.float32)

    print(f"[INFO] Combined dataset: N={len(y)}, D={X.shape[1]}")
    print(f"[INFO] y stats: mean={float(np.mean(y)):.4f} std={float(np.std(y)):.4f} min={float(np.min(y)):.4f} max={float(np.max(y)):.4f}")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta_all,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", Ridge(alpha=args.alpha, random_state=args.seed)),
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred_clip = np.clip(pred, 0.0, 1.0)

    mse = mean_squared_error(y_test, pred_clip)
    mae = mean_absolute_error(y_test, pred_clip)
    r2 = r2_score(y_test, pred_clip)

    print(f"[RESULT] Test N={len(y_test)}")
    print(f"[RESULT] MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.6f}")
    print(f"[RESULT] y_test mean={float(np.mean(y_test)):.4f}  pred mean={float(np.mean(pred_clip)):.4f}")

    # save a tiny diagnostic sample list (first 50)
    diag = []
    for (ds, qid, kc_idx), yt, yp in zip(meta_test[:50], y_test[:50], pred_clip[:50]):
        diag.append({"dataset": ds, "qid": qid, "kc_idx": kc_idx, "y": float(yt), "pred": float(yp)})

    payload = {
        "model": model,
        "embedding_key": args.embedding_key,
        "alpha": args.alpha,
        "seed": args.seed,
        "test_size": args.test_size,
        "min_models": args.min_models,
        "datasets_used": [d.name for d in dataset_dirs],
        "diagnostic_examples": diag,
    }

    joblib.dump(payload, args.out_model)
    print(f"[INFO] Saved model bundle to: {args.out_model}")


if __name__ == "__main__":
    main()
