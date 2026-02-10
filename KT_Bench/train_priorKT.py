import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from model_priorKT import PriorKT, initialize
from utils import Logger, Saver, Metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sequences(df: pd.DataFrame):
    if "timestamp" in df.columns:
        df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    else:
        df = df.sort_values(["user_id"], kind="mergesort")

    seqs = []
    for uid, u_df in df.groupby("user_id", sort=False):
        items = u_df["item_id"].values
        labels = u_df["correct"].values

        if len(items) == 0:
            continue

        it = items.copy()
        lb = labels.copy()
        seqs.append((it, lb, uid))

    return seqs



def split_train_val_ensure_all_items(
    seqs,
    all_items_set,
    train_ratio: float = 0.8,
    seed: int = 0,
):
    rng = random.Random(seed)
    seqs = list(seqs)
    rng.shuffle(seqs)

    n_train = int(train_ratio * len(seqs))
    train = seqs[:n_train]
    val = seqs[n_train:]

    def items_in(seqs_):
        s = set()
        for it, _, _ in seqs_:
            s.update(it.tolist())
        return s

    train_items = items_in(train)
    missing = set(all_items_set) - train_items

    if missing:
        moved = 0
        i = 0
        while missing and i < len(val):
            it, lb, uid = val[i]
            covers = missing.intersection(it.tolist())
            if covers:
                train.append((it, lb, uid))
                train_items.update(it.tolist())
                missing = set(all_items_set) - train_items
                val.pop(i)
                moved += 1
            else:
                i += 1
        if missing:
            print(f"[WARN] Could not cover all missing items in train split. Remaining: {len(missing)}")
        else:
            print(f"[INFO] Moved {moved} sequences from val->train to cover all items.")

    return train, val


class BagDataset(Dataset):
    def __init__(self, seqs):
        self.samples = []
        
        # Unroll sequences into (History, Target, Label) samples
        
        print(f"Processing {len(seqs)} sequences into Bag samples...")
        for items, labels, uid in tqdm(seqs):
            seq_len = len(items)
            for t in range(seq_len):
                target = items[t]
                label = labels[t]
                
                hist_idx = items[0:t]
                hist_val = labels[0:t]
                
                # Transform labels: 0 -> -1.0, 1 -> 1.0
                # This logic puts the score transformation in the dataset
                if len(hist_val) > 0:
                    hist_val_transformed = (hist_val.astype(np.float32) * 2.0) - 1.0
                else:
                    hist_val_transformed = np.array([], dtype=np.float32)

                self.samples.append((hist_idx, hist_val_transformed, target, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_bag_fn(batch):
    # batch: list of (h_idx, h_val, target, label)
    
    h_idx_list = []
    h_val_list = []
    target_list = []
    label_list = []

    for h_i, h_v, t, l in batch:
        # Shift Item IDs by +1 because 0 is padding in Model
        # Data IDs are 0-based. Model IDs: 0=Pad, 1=Item0, ...
        h_idx_list.append(torch.tensor(h_i + 1, dtype=torch.long))
        h_val_list.append(torch.tensor(h_v, dtype=torch.float))
        target_list.append(t + 1)
        label_list.append(l)

    # Pad with 0
    h_idx_pad = pad_sequence(h_idx_list, batch_first=True, padding_value=0)
    # Pad values with 0 (neutral influence)
    h_val_pad = pad_sequence(h_val_list, batch_first=True, padding_value=0.0)

    targets = torch.tensor(target_list, dtype=torch.long)
    labels = torch.tensor(label_list, dtype=torch.float)

    return h_idx_pad, h_val_pad, targets, labels


def train_one_epoch(
    model: PriorKT,
    dataloader: DataLoader,
    optimizer,
    criterion,
    grad_clip: float,
    use_amp: bool,
):
    model.train()
    metrics = Metrics()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for h_idx, h_val, targets, labels in tqdm(dataloader, leave=False, desc="Train"):
        h_idx = h_idx.cuda(non_blocking=True)
        h_val = h_val.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(h_idx, h_val, targets)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        metrics.store({"loss/train": loss.item()})

    return metrics


@torch.no_grad()
def eval_epoch(model: PriorKT, dataloader: DataLoader, use_amp: bool):
    model.eval()
    metrics = Metrics()

    all_probs = []
    all_labels = []

    for h_idx, h_val, targets, labels in tqdm(dataloader, leave=False, desc="Eval"):
        h_idx = h_idx.cuda(non_blocking=True)
        h_val = h_val.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(h_idx, h_val, targets)
            probs = torch.sigmoid(logits)
        
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    if all_probs:
        probs_cat = torch.cat(all_probs)
        labels_cat = torch.cat(all_labels)
        try:
            auc = roc_auc_score(labels_cat.numpy(), probs_cat.numpy())
        except ValueError:
            auc = 0.5
        metrics.store({"auc/val": float(auc)})

    return metrics


@torch.no_grad()
def predict_test(model: PriorKT, dataloader: DataLoader, use_amp: bool):
    model.eval()
    preds_all = []

    for h_idx, h_val, targets, _ in tqdm(dataloader, desc="Predicting"):
        h_idx = h_idx.cuda(non_blocking=True)
        h_val = h_val.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(h_idx, h_val, targets)
            probs = torch.sigmoid(logits)
            preds_all.extend(probs.cpu().numpy().tolist())

    return np.array(preds_all, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline KT (Bag of Items).")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="runs/baseline")
    parser.add_argument("--savedir", type=str, default="save/baseline")

    parser.add_argument("--rank", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_epochs", type=int, default=50)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required."

    set_seed(args.seed)
    use_amp = (not args.no_amp)

    data_dir = os.path.join("data", args.dataset)
    train_path = os.path.join(data_dir, "preprocessed_data_train.csv")
    test_path = os.path.join(data_dir, "preprocessed_data_test.csv")

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    test_df["_orig_idx"] = np.arange(len(test_df))

    full_max_item = max(train_df["item_id"].max(), test_df["item_id"].max())
    num_items = int(full_max_item + 1)
    valid_rank = min(num_items,args.rank)
    print(f"[INFO] Dataset: {args.dataset}, Num Items: {num_items}")

    # Initialize Prior
    # We pass the train path to compute item priors from first attempts
    init_artifacts = initialize(
        train_path=train_path,
        num_items=num_items,
        user_col="user_id",
        item_col="item_id",
        label_col="correct",
        sep="\t",
    )

    # Prepare Sequences
    print("[INFO] Loading Sequences...")
    seqs_all = get_sequences(train_df)
    all_items_set = set(train_df["item_id"].unique().tolist())
    
    print("[INFO] Splitting Train/Val...")
    train_seqs, val_seqs = split_train_val_ensure_all_items(
        seqs_all, all_items_set, train_ratio=0.8, seed=args.seed
    )

    # Prepare Datasets
    # BagDataset converts sequences to step-wise (history, target) pairs
    train_ds = BagDataset(train_seqs)
    val_ds = BagDataset(val_seqs)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_bag_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_bag_fn,
        num_workers=4,
        pin_memory=True
    )

    # Model Setup
    model = PriorKT(
        pi=init_artifacts.pi,
        rank=valid_rank,
        pad_id=0, # Model internal padding is 0
        init_embed_std=1e-3,
    ).cuda()

    print(f"[INFO] Model: {model.diagnostics()}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    param_str = f"{args.dataset}_bag_r{valid_rank}_lr{args.lr}"
    logger = Logger(os.path.join(args.logdir, param_str))
    saver = Saver(args.savedir, param_str)

    best_auc = -1.0
    no_improve = 0

    print("[INFO] Start Training...")
    for epoch in range(args.num_epochs):
        tr_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            grad_clip=args.grad_clip, use_amp=use_amp
        )
        va_metrics = eval_epoch(
            model, val_loader, use_amp=use_amp
        )

        avg = {}
        avg.update(tr_metrics.average())
        avg.update(va_metrics.average())
        avg["epoch"] = epoch

        logger.log_scalars(avg, epoch)
        print(f"[Epoch {epoch}] " + ", ".join([f"{k}={v:.5f}" for k, v in avg.items() if k != "epoch"]))

        auc_val = avg.get("auc/val", 0.0)
        if auc_val > best_auc + args.min_delta:
            best_auc = auc_val
            no_improve = 0
            saver.save(best_auc, model)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[EarlyStopping] No improvement for {args.patience} epochs. Stop.")
                break

    logger.close()

    # Inference on Test
    print("[INFO] Running Inference on Test Set...")

    test_df_sorted = test_df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    test_seqs = get_sequences(test_df_sorted)
    
    # BagDataset works for Test too (creates (history, target) pairs)
    test_ds = BagDataset(test_seqs)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_bag_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_preds = predict_test(
        model, test_loader, use_amp=use_amp
    )

    # Re-align predictions to dataframe
    # Because BagDataset unrolls sequences, the order matches exactly the row order 
    # of the items in 'test_seqs' which was built from 'test_df_sorted'.
    # We verify lengths match.
    
    if len(test_preds) != len(test_df_sorted):
        # Fallback alignment if get_sequences skipped empty users or similar edge cases.
        # But get_sequences iterates groupby which matches unique users.
        # If any user had 0 items, they are skipped.
        print(f"[WARN] Length mismatch: preds={len(test_preds)} vs df={len(test_df_sorted)}")
    
    # Assign to sorted df
    test_df_sorted["BASELINE"] = test_preds
    
    # Restore original order
    test_out = test_df_sorted.sort_values("_orig_idx", kind="mergesort").drop(columns=["_orig_idx"])
    test_out.to_csv(test_path, sep="\t", index=False)
    
    if "correct" in test_out.columns:
        try:
            auc_test = roc_auc_score(test_out["correct"].values, test_out["BASELINE"].values)
            print(f"auc_test = {auc_test:.5f}")
        except:
            print("Could not compute test AUC.")


if __name__ == "__main__":
    main()