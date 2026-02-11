import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------
DATA_TSV = "preprocessed_data.csv"                 # tab-separated
KC_JSON = "output_with_embedding_octen.json"       # has KCs, prior
CLUSTER_JSON = "kc_labels/clustering_k50.json"     # from clustering.py

# Petrov (2006) Eq.4 with d=0.5, k=1, gain k=1
K_GAIN = 1.0

# Time window: from user's first timestamp to +30 days
WINDOW_DAYS = 30

# Sampling resolution
STEP_HOURS = 6

# User selection
MIN_SPAN_DAYS = 30
PICK_USER_MODE = "most_records"  # "most_records" or "random"
RANDOM_SEED = 0

# Item selection within user
PICK_ITEM_MODE = "most_attempted"  # "most_attempted" or "random"


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

def kc_prob(prior: float, n: int, t_recent: float, t_life: float) -> float:
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

# skill_id -> {"prior":[...], "KCs":[...]}
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

# clustering output: skill_to_clusters aligns with original KCs order
skill_to_clusters = {str(k): [int(x) for x in v] for k, v in cl["skill_to_clusters"].items()}

# filter rows to skills existing in both maps
valid_skills = set(skill_to_clusters.keys()) & set(skill_to_prior_list.keys())
df = df[df["skill_id"].isin(valid_skills)].copy()
df.sort_values(["user_id", "timestamp"], inplace=True)

# ---------------------------
# Build cluster-level prior by aggregating original priors
# (mean over all occurrences across skills)
# ---------------------------
cluster_prior_sum = defaultdict(float)
cluster_prior_cnt = defaultdict(int)

for sid in valid_skills:
    pri_list = skill_to_prior_list[sid]
    clu_list = skill_to_clusters[sid]
    if len(pri_list) != len(clu_list):
        raise ValueError(f"Cluster alignment mismatch at skill_id={sid}: len(prior)={len(pri_list)} vs len(clusters)={len(clu_list)}")

    for p, c in zip(pri_list, clu_list):
        if c < 0:
            continue
        cluster_prior_sum[c] += float(p)
        cluster_prior_cnt[c] += 1

cluster_prior = {}
for c, s in cluster_prior_sum.items():
    cluster_prior[c] = s / max(cluster_prior_cnt[c], 1)

if len(cluster_prior) == 0:
    raise RuntimeError("cluster_prior is empty. Check clustering_k50.json and prior field availability.")


# ---------------------------
# Filter users: span >= 30 days
# ---------------------------
user_span = df.groupby("user_id")["timestamp"].agg(["min", "max", "count"]).reset_index()
user_span["span_days"] = (user_span["max"] - user_span["min"]) / (60 * 60 * 24)

eligible = user_span[user_span["span_days"] >= MIN_SPAN_DAYS].copy()
if len(eligible) == 0:
    raise RuntimeError("No users with span >= 30 days. Lower MIN_SPAN_DAYS or check timestamps.")

if PICK_USER_MODE == "most_records":
    eligible.sort_values("count", ascending=False, inplace=True)
    picked_user = str(eligible.iloc[0]["user_id"])
else:
    rng = np.random.default_rng(RANDOM_SEED)
    picked_user = str(rng.choice(eligible["user_id"].values))

udf = df[df["user_id"] == picked_user].copy()
udf.sort_values("timestamp", inplace=True)

t0 = float(udf["timestamp"].min())
t_end = t0 + WINDOW_DAYS * 24 * 60 * 60
udf_first_month = udf[udf["timestamp"] <= t_end].copy()

print(f"[INFO] picked_user={picked_user}")
print(f"[INFO] user_span_days={(udf['timestamp'].max()-udf['timestamp'].min())/(60*60*24):.2f}")
print(f"[INFO] first_month_events={len(udf_first_month)}/{len(udf)}")


# ---------------------------
# Pick an item for that user
# ---------------------------
item_counts = udf_first_month["item_id"].value_counts()
if len(item_counts) == 0:
    item_counts = udf["item_id"].value_counts()
if len(item_counts) == 0:
    raise RuntimeError("Picked user has no item interactions.")

if PICK_ITEM_MODE == "most_attempted":
    picked_item = str(item_counts.index[0])
else:
    rng = np.random.default_rng(RANDOM_SEED)
    picked_item = str(rng.choice(item_counts.index.values))

print(f"[INFO] picked_item={picked_item}, attempts_in_first_month={int((udf_first_month['item_id']==picked_item).sum())}")


# ---------------------------
# Simulation (cluster-level memory)
# ---------------------------
# state per cluster id: n, t_first, t_last
cluster_state = {}

events = udf[["timestamp", "skill_id", "item_id", "correct"]].to_records(index=False)

step_sec = STEP_HOURS * 60 * 60
grid = np.arange(t0, t_end + 1, step_sec, dtype=np.float64)

eidx = 0
num_events = len(events)

pred_p = []

def clusters_for_skill(sid: str) -> list[int]:
    # deduplicate per skill attempt: exposure counted once per cluster
    cls = [c for c in skill_to_clusters[sid] if c >= 0]
    if not cls:
        return []
    return sorted(set(cls))

def item_prob_at_time(query_t: float, item_id: str) -> float:
    # choose a skill_id for that item: most recent occurrence up to query_t, else earliest
    sub = udf[(udf["item_id"] == item_id) & (udf["timestamp"] <= query_t)]
    if len(sub) > 0:
        sid = str(sub.iloc[-1]["skill_id"])
    else:
        sid = str(udf[udf["item_id"] == item_id].iloc[0]["skill_id"])

    cls = clusters_for_skill(sid)
    if len(cls) == 0:
        return 1.0

    probs = []
    for c in cls:
        prior = float(cluster_prior.get(c, 0.5))  # fallback if unseen
        st = cluster_state.get(c)
        if st is None:
            probs.append(prior)
        else:
            n = st["n"]
            t_recent = query_t - st["t_last"]
            t_life = query_t - st["t_first"]
            probs.append(kc_prob(prior=prior, n=n, t_recent=t_recent, t_life=t_life))

    # aggregation: mean of cluster recall probabilities
    return float(np.mean(probs))

for qt in grid:
    # update cluster memory with all events up to qt
    while eidx < num_events and float(events[eidx][0]) <= qt:
        ts, sid, it, y = events[eidx]
        sid = str(sid)
        cls = clusters_for_skill(sid)

        # update each cluster once
        for c in cls:
            st = cluster_state.get(c)
            if st is None:
                cluster_state[c] = {"n": 1, "t_first": float(ts), "t_last": float(ts)}
            else:
                st["n"] += 1
                st["t_last"] = float(ts)

        eidx += 1

    pred_p.append(item_prob_at_time(qt, picked_item))

pred_p = np.asarray(pred_p, dtype=np.float64)

attempts = udf_first_month[udf_first_month["item_id"] == picked_item].copy()
attempt_times = attempts["timestamp"].to_numpy(dtype=np.float64)
attempt_y = attempts["correct"].to_numpy(dtype=np.int32)

x_days = (grid - t0) / (60 * 60 * 24)
attempt_x_days = (attempt_times - t0) / (60 * 60 * 24)


# ---------------------------
# Plot
# ---------------------------
plt.figure()
plt.plot(x_days, pred_p, label="Predicted P(correct) (cluster50 BLA+prior, mean agg)")
if len(attempt_times) > 0:
    plt.scatter(attempt_x_days, attempt_y, marker="x", label="Actual correctness (attempts)")

plt.title(f"User {picked_user} · Item {picked_item} · First {WINDOW_DAYS} days (cluster K=50)")
plt.xlabel("Days since user started")
plt.ylabel("Probability / Outcome")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(f"User {picked_user} · Item {picked_item} · First {WINDOW_DAYS} days (cluster K=50).png", dpi=700)
plt.show()