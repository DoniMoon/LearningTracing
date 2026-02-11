import json
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------
DATA_TSV = "preprocessed_data.csv"                 # tab-separated
KC_JSON = "output_with_embedding_octen.json"       # has prior
CLUSTER_JSON = "kc_labels/clustering_k50.json"     # from clustering.py

OUT_PNG = "rq2_2_mean_curve_top100_3items_30d_cluster50.png"
DPI = 1200

WINDOW_DAYS = 30
STEP_HOURS = 6

MIN_SPAN_DAYS = 30          # user must have >=30 days span overall
TOP_USERS = 100             # pick top by interactions in first 30 days

NUM_ITEMS = 3               # keep 3 items
PICK_ITEMS_MODE = "most_common"   # keep automatic selection

# Petrov Eq.4 with d=0.5, k=1 and gain k=1
K_GAIN = 1.0


# ---------------------------
# Math utils
# ---------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def safe_dt(dt: float, eps: float = 1.0) -> float:
    return max(float(dt), eps)

def bla_petrov_eq4_k1(n: int, t_recent: float, t_life: float, eps: float = 1.0) -> float:
    """
    Petrov (2006) Eq.4, d=0.5, k=1:
      B ≈ ln[ 1/sqrt(t_recent) + 2(n-1)/(sqrt(t_life)+sqrt(t_recent)) ]
    """
    if n <= 0:
        raise ValueError("n must be >= 1 for observed KC")

    t_recent = safe_dt(t_recent, eps)
    t_life = safe_dt(t_life, eps)

    term_recent = 1.0 / math.sqrt(t_recent)
    if n == 1:
        s = term_recent
    else:
        s = term_recent + (2.0 * (n - 1)) / (math.sqrt(t_life) + math.sqrt(t_recent))
    return math.log(s)

def cluster_prob(prior: float, n: int, t_recent: float, t_life: float) -> float:
    if n <= 0:
        return float(prior)
    B = bla_petrov_eq4_k1(n=n, t_recent=t_recent, t_life=t_life)
    return sigmoid(K_GAIN * B)


# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_TSV, sep="\t")
df["user_id"] = df["user_id"].astype(str)
df["skill_id"] = df["skill_id"].astype(str)
df["item_id"] = df["item_id"].astype(str)
df["timestamp"] = df["timestamp"].astype(float)
df["correct"] = df["correct"].astype(float).astype(int)

with open(KC_JSON, "r", encoding="utf-8") as f:
    kc_data = json.load(f)

# skill_id -> prior list
skill_to_prior_list = {}
for skill_id, obj in kc_data.items():
    sid = str(skill_id)
    pri = obj.get("prior", None)
    kcs = obj.get("KCs", None)
    if pri is None or kcs is None:
        continue
    if len(pri) != len(kcs):
        raise ValueError(f"Length mismatch at skill_id={sid}: len(KCs)={len(kcs)} vs len(prior)={len(pri)}")
    skill_to_prior_list[sid] = [float(p) for p in pri]

with open(CLUSTER_JSON, "r", encoding="utf-8") as f:
    cl = json.load(f)

skill_to_clusters = {str(k): [int(x) for x in v] for k, v in cl["skill_to_clusters"].items()}

valid_skills = set(skill_to_clusters.keys()) & set(skill_to_prior_list.keys())
df = df[df["skill_id"].isin(valid_skills)].copy()
df.sort_values(["user_id", "timestamp"], inplace=True)

# item_id -> 대표 skill_id
item_skill_mode = (
    df.groupby(["item_id", "skill_id"]).size().reset_index(name="cnt")
      .sort_values(["item_id", "cnt"], ascending=[True, False])
      .drop_duplicates("item_id")
)
item_to_skill = dict(zip(item_skill_mode["item_id"], item_skill_mode["skill_id"]))

def clusters_for_skill(sid: str) -> list[int]:
    cls = [c for c in skill_to_clusters[sid] if c >= 0]
    if not cls:
        return []
    return sorted(set(cls))  # dedup per attempt


# ---------------------------
# cluster-level prior
# ---------------------------
cluster_prior_sum = defaultdict(float)
cluster_prior_cnt = defaultdict(int)

for sid in valid_skills:
    pri_list = skill_to_prior_list[sid]
    clu_list = skill_to_clusters[sid]
    if len(pri_list) != len(clu_list):
        raise ValueError(f"Alignment mismatch at skill_id={sid}")
    for p, c in zip(pri_list, clu_list):
        if c < 0:
            continue
        cluster_prior_sum[c] += float(p)
        cluster_prior_cnt[c] += 1

cluster_prior = {c: cluster_prior_sum[c] / max(cluster_prior_cnt[c], 1) for c in cluster_prior_sum.keys()}


# ---------------------------
# Eligible users: span >= 30 days
# and pick top-100 by interactions within first 30 days
# ---------------------------
user_agg = df.groupby("user_id")["timestamp"].agg(["min", "max", "count"]).reset_index()
user_agg["span_days"] = (user_agg["max"] - user_agg["min"]) / (60 * 60 * 24)

eligible = user_agg[user_agg["span_days"] >= MIN_SPAN_DAYS].copy()
if len(eligible) == 0:
    raise RuntimeError("No users with span >= 30 days.")

# compute first-30d interactions for each eligible user
first_window_end = {}
for _, r in eligible.iterrows():
    uid = str(r["user_id"])
    t0 = float(r["min"])
    first_window_end[uid] = t0 + WINDOW_DAYS * 24 * 60 * 60

# count interactions within first 30 days
first30_counts = []
for uid in eligible["user_id"].astype(str).tolist():
    t_end = first_window_end[uid]
    cnt = int((df["user_id"].eq(uid) & (df["timestamp"] <= t_end)).sum())
    first30_counts.append((uid, cnt))

first30_counts.sort(key=lambda x: x[1], reverse=True)
picked_users = [uid for uid, cnt in first30_counts[:TOP_USERS] if cnt > 0]

if len(picked_users) == 0:
    raise RuntimeError("No picked users after applying top-100 by first-30d interactions.")

print(f"[INFO] eligible_users={len(eligible)}, picked_top_users={len(picked_users)}")
print(f"[INFO] top_user_first30_counts (head): {first30_counts[:5]}")


# ---------------------------
# Pick 3 target items among picked users in first 30 days
# ---------------------------
counts = Counter()
for uid in picked_users:
    udf = df[df["user_id"] == uid]
    t0 = float(udf["timestamp"].min())
    t_end = t0 + WINDOW_DAYS * 24 * 60 * 60
    items = udf[udf["timestamp"] <= t_end]["item_id"].astype(str).unique()
    for it in items:
        counts[it] += 1

target_items = []
for it, _c in counts.most_common():
    if it not in item_to_skill:
        continue
    sid = str(item_to_skill[it])
    cls = clusters_for_skill(sid)
    if len(cls) == 0:
        continue
    target_items.append(it)
    if len(target_items) >= NUM_ITEMS:
        break

if len(target_items) < NUM_ITEMS:
    raise RuntimeError("Not enough target items found for 3-item plot among picked users.")

print(f"[INFO] target_items={target_items}")


# ---------------------------
# Precompute required clusters per item
# ---------------------------
item_req_clusters = {}
for it in target_items:
    sid = str(item_to_skill[it])
    cls = clusters_for_skill(sid)
    if len(cls) == 0:
        raise RuntimeError(f"Item {it} has empty required cluster list.")
    item_req_clusters[it] = cls
    print(f"[INFO] item={it} req_clusters={len(cls)} (skill={sid})")


# ---------------------------
# Simulation engine
# ---------------------------
step_sec = STEP_HOURS * 60 * 60
grid_rel = np.arange(0, WINDOW_DAYS * 24 * 60 * 60 + 1, step_sec, dtype=np.float64)
x_days = grid_rel / (60 * 60 * 24)

def simulate_user_curves(uid: str) -> dict:
    """
    For user uid, simulate first 30 days memory and return curves for each target item.
    """
    udf = df[df["user_id"] == uid].copy()
    udf.sort_values("timestamp", inplace=True)

    t0 = float(udf["timestamp"].min())
    t_end = t0 + WINDOW_DAYS * 24 * 60 * 60

    uev = udf[udf["timestamp"] <= t_end][["timestamp", "skill_id"]].to_records(index=False)

    state = {}  # cluster -> {n,t_first,t_last}
    eidx = 0
    num_e = len(uev)

    out = {it: [] for it in target_items}

    for rel_t in grid_rel:
        qt = t0 + float(rel_t)

        while eidx < num_e and float(uev[eidx][0]) <= qt:
            ts, sid = uev[eidx]
            sid = str(sid)
            cls = clusters_for_skill(sid)
            for c in cls:
                st = state.get(c)
                if st is None:
                    state[c] = {"n": 1, "t_first": float(ts), "t_last": float(ts)}
                else:
                    st["n"] += 1
                    st["t_last"] = float(ts)
            eidx += 1

        for it in target_items:
            req = item_req_clusters[it]
            probs = []
            for c in req:
                prior = float(cluster_prior.get(c, 0.5))
                st = state.get(c)
                if st is None:
                    probs.append(prior)
                else:
                    n = st["n"]
                    t_recent = qt - st["t_last"]
                    t_life = qt - st["t_first"]
                    probs.append(cluster_prob(prior=prior, n=n, t_recent=t_recent, t_life=t_life))
            out[it].append(float(np.mean(probs)))

    return out


# ---------------------------
# Aggregate over top-100 users
# ---------------------------
item_all = {it: [] for it in target_items}

for i, uid in enumerate(picked_users, 1):
    curves = simulate_user_curves(uid)
    for it in target_items:
        item_all[it].append(curves[it])
    if i % 25 == 0:
        print(f"[INFO] processed users: {i}/{len(picked_users)}")

item_stats = {}
for it in target_items:
    arr = np.asarray(item_all[it], dtype=np.float64)  # (U, T)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    se = std / max(math.sqrt(arr.shape[0]), 1.0)
    item_stats[it] = (mean, se, arr.shape[0])


# ---------------------------
# Plot + save
# ---------------------------
plt.figure()

for it in target_items:
    mean, se, U = item_stats[it]
    plt.plot(x_days, mean, label=f"item {it} (U={U})")
    plt.fill_between(x_days, mean - se, mean + se, alpha=0.2)

plt.title(f"First {WINDOW_DAYS} days · Top-{TOP_USERS} by first-{WINDOW_DAYS}d interactions · cluster K=50")
plt.xlabel("Days since each user started")
plt.ylabel("Predicted P(correct)")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
print(f"[OK] saved {OUT_PNG} (dpi={DPI})")

plt.show()
