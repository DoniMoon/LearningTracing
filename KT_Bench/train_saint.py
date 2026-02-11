import argparse
import os
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from model_saint import SAINT

class Metrics:
    def __init__(self):
        self.metrics = {}

    def store(self, key_value_dict):
        for k, v in key_value_dict.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)

    def average(self):
        return {k: np.mean(v) for k, v in self.metrics.items()}

class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.log_file = open(os.path.join(logdir, 'log.txt'), 'w')

    def log_scalars(self, scalar_dict, step):
        log_str = f"Step {step} | " + " | ".join([f"{k}: {v:.4f}" for k, v in scalar_dict.items()])
        self.log_file.write(log_str + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

class Saver:
    def __init__(self, savedir, param_str, patience=5):
        self.savedir = savedir
        self.param_str = param_str
        self.patience = patience
        self.counter = 0
        self.best_auc = -1
        
        if not os.path.exists(savedir):
            os.makedirs(savedir)

    def save(self, auc, model):
        if auc > self.best_auc:
            self.best_auc = auc
            self.counter = 0 # 성능 향상 시 카운터 리셋
            torch.save(model.state_dict(), os.path.join(self.savedir, f'best_model_{self.param_str}.pth'))
            return False 
        else:
            self.counter += 1
            print(f"  [Saver] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("  [Saver] Early stopping triggered.")
                return True
            return False

def get_data(df, max_length, train_split=0.8, randomize=True):# Group by user_id (User ID)
    grouped = df.groupby("user_id")
    
    item_seqs = [torch.tensor(u_df["item_id"].values, dtype=torch.long) + 1 
                 for _, u_df in grouped]
    
    skill_seqs = [torch.tensor(u_df["skill_id"].values, dtype=torch.long) + 1 
                  for _, u_df in grouped]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long) 
              for _, u_df in grouped]

    def chunk(seq_list, chunk_size):
        chunked = []
        for seq in seq_list:
            num_chunks = (len(seq) - 1) // chunk_size + 1
            for i in range(num_chunks):
                chunked.append(seq[i*chunk_size : (i+1)*chunk_size])
        return chunked

    item_chunks = chunk(item_seqs, max_length)
    skill_chunks = chunk(skill_seqs, max_length)
    label_chunks = chunk(labels, max_length)

    response_chunks = []
    for label_seq in label_chunks:
        shifted = torch.cat((torch.zeros(1, dtype=torch.long), label_seq[:-1] + 1))
        response_chunks.append(shifted)

    data = list(zip(item_chunks, skill_chunks, response_chunks, label_chunks))
    
    if randomize:
        shuffle(data)

    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data

def prepare_batches(data, batch_size, randomize=True):
    if randomize:
        shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        item_seqs, skill_seqs, response_seqs, label_seqs = zip(*batch)
        
        item_pad = pad_sequence(item_seqs, batch_first=True, padding_value=0)
        skill_pad = pad_sequence(skill_seqs, batch_first=True, padding_value=0)
        
        response_pad = pad_sequence(response_seqs, batch_first=True, padding_value=0)
        
        label_pad = pad_sequence(label_seqs, batch_first=True, padding_value=-1)
        
        batches.append((item_pad, skill_pad, response_pad, label_pad))

    return batches

def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    
    if len(preds) == 0:
        return 0.5
        
    if len(torch.unique(labels)) == 1:  # Only one class
        return accuracy_score(labels, preds.round())
    else:
        return roc_auc_score(labels, preds)

def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)

def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, grad_clip):
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size, randomize=False)

        model.train()
        for item_ids, skill_ids, responses, labels in train_batches:
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            responses = responses.to(device)
            labels = labels.to(device)

            preds = model(item_ids, skill_ids, responses)
            
            loss = compute_loss(preds, labels, criterion)
            
            preds_prob = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds_prob, labels.cpu())

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})

            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)

        model.eval()
        for item_ids, skill_ids, responses, labels in val_batches:
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            responses = responses.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                preds = model(item_ids, skill_ids, responses)
                preds_prob = torch.sigmoid(preds).cpu()
            
            val_auc = compute_auc(preds_prob, labels.cpu())
            metrics.store({'auc/val': val_auc})

        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation AUC: {average_metrics.get('auc/val', 0):.4f}")
        
        stop_training = saver.save(average_metrics.get('auc/val', 0), model)
        if stop_training:
            print("Training stopped early.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAINT.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset folder name')
    parser.add_argument('--logdir', type=str, default='runs/saint')
    parser.add_argument('--savedir', type=str, default='save/saint')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2, help='Number of encoder/decoder layers')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--max_pos', type=int, default=200)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    args = parser.parse_args()

    data_path = os.path.join('data', args.dataset, 'preprocessed_data.csv')
    train_path = os.path.join('data', args.dataset, 'preprocessed_data_train.csv')
    test_path = os.path.join('data', args.dataset, 'preprocessed_data_test.csv')

    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, sep="\t")
        train_data, val_data = get_data(train_df, args.max_length)
    elif os.path.exists(data_path):
        full_df = pd.read_csv(data_path, sep="\t")
        train_data, val_data = get_data(full_df, args.max_length)
    else:
        raise FileNotFoundError(f"Data not found in data/{args.dataset}/")

    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, sep="\t")
        train_data, val_data = get_data(train_df, args.max_length)
        
        if os.path.exists(data_path):
            full_df = pd.read_csv(data_path, sep="\t")
            num_items = int(full_df["item_id"].max() + 1)
            num_skills = int(full_df["skill_id"].max() + 1)
        else:
            test_df_temp = pd.read_csv(test_path, sep="\t")
            max_item_train = train_df["item_id"].max()
            max_item_test = test_df_temp["item_id"].max()
            num_items = int(max(max_item_train, max_item_test) + 1)
            
            max_skill_train = train_df["skill_id"].max()
            max_skill_test = test_df_temp["skill_id"].max()
            num_skills = int(max(max_skill_train, max_skill_test) + 1)
    elif os.path.exists(data_path):
        full_df = pd.read_csv(data_path, sep="\t")
        train_data, val_data = get_data(full_df, args.max_length)
        num_items = int(full_df["item_id"].max() + 1)
        num_skills = int(full_df["skill_id"].max() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SAINT(num_items=num_items, 
                  num_skills=num_skills, 
                  embed_size=args.embed_size, 
                  num_layers=args.num_layers, 
                  num_heads=args.num_heads, 
                  max_pos=args.max_pos, 
                  drop_prob=args.drop_prob).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"Starting training on {device}...")
    print(f"Dataset: {args.dataset} | Items: {num_items} | Skills: {num_skills}")
    print(f"Early Stopping Patience: {args.patience}")
    
    param_str = (f'{args.dataset}_saint_'
                 f'L{args.num_layers}_H{args.num_heads}_D{args.embed_size}_'
                 f'bs{args.batch_size}')
                 
    logger = Logger(os.path.join(args.logdir, param_str))
    
    saver = Saver(args.savedir, param_str, patience=args.patience)

    train(train_data, val_data, model, optimizer, logger, saver, 
          args.num_epochs, args.batch_size, args.grad_clip)

    logger.close()
    
    if os.path.exists(test_path):
        print("Predicting on Test set...")
        test_df = pd.read_csv(test_path, sep="\t")
        test_data, _ = get_data(test_df, args.max_length, train_split=1.0, randomize=False)
        test_batches = prepare_batches(test_data, args.batch_size, randomize=False)
        
        test_preds = np.empty(0)
        
        # Load best model for testing
        best_model_path = os.path.join(args.savedir, f'best_model_{param_str}.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing.")
        
        model.eval()
        for item_ids, skill_ids, responses, labels in test_batches:
            item_ids = item_ids.to(device)
            skill_ids = skill_ids.to(device)
            responses = responses.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                preds = model(item_ids, skill_ids, responses)
                preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
                test_preds = np.concatenate([test_preds, preds])
        
        test_df["SAINT"] = test_preds
        test_df.to_csv(os.path.join('data', args.dataset, 'preprocessed_data_test_saint.csv'), sep="\t", index=False) # seperate file saving due to long training time of SAINT.
        print("auc_test = ", roc_auc_score(test_df["correct"], test_preds))