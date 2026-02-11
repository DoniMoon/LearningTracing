import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

def attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)
        self.out_proj = nn.Linear(total_size, total_size)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length = query.shape[:2]

        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        out, self.prob_attn = attention(query, key, value, mask, self.dropout)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.total_size)
        return self.out_proj(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        m = self.sublayer[1](m, lambda m: self.src_attn(m, memory, memory, src_mask))
        
        return self.sublayer[2](m, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class SAINT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_layers, num_heads,
                 max_pos, drop_prob):
        super(SAINT, self).__init__()
        self.embed_size = embed_size
        
        self.item_embeds = nn.Embedding(num_items + 1, embed_size, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size, padding_idx=0)
        self.pos_embeds = nn.Embedding(max_pos, embed_size)
        self.response_embeds = nn.Embedding(3, embed_size) 

        c = copy.deepcopy
        attn = MultiHeadedAttention(embed_size, num_heads, drop_prob)
        ff = PositionwiseFeedForward(embed_size, embed_size * 4, drop_prob)
        
        self.encoder_layers = clone(EncoderLayer(embed_size, c(attn), c(ff), drop_prob), num_layers)
        self.decoder_layers = clone(DecoderLayer(embed_size, c(attn), c(attn), c(ff), drop_prob), num_layers)
        
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(embed_size, 1)

    def forward(self, item_ids, skill_ids, responses):
        device = item_ids.device
        batch_size, seq_len = item_ids.shape
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embeds(pos_ids)
        
        enc_in = self.item_embeds(item_ids) + self.skill_embeds(skill_ids) + pos_emb
        enc_in = self.dropout(enc_in)
        
        enc_mask = future_mask(seq_len).to(device)
        
        for layer in self.encoder_layers:
            enc_in = layer(enc_in, enc_mask)
            
        enc_out = self.norm(enc_in)

        dec_in = self.response_embeds(responses) + pos_emb
        dec_in = self.dropout(dec_in)
        
        dec_mask = future_mask(seq_len).to(device)
        
        for layer in self.decoder_layers:
            dec_in = layer(dec_in, enc_out, src_mask=enc_mask, tgt_mask=dec_mask)
            
        dec_out = self.norm(dec_in)
        
        output = self.out_proj(dec_out)
        return output

if __name__ == "__main__":
    bs = 32
    seq = 50
    item_n = 100
    skill_n = 20
    
    i = torch.randint(1, item_n, (bs, seq))
    s = torch.randint(1, skill_n, (bs, seq))
    r = torch.randint(1, 3, (bs, seq)) # 1 or 2
    
    model = SAINT(num_items=item_n, num_skills=skill_n, embed_size=64, 
                  num_layers=2, num_heads=4, max_pos=100, drop_prob=0.1)
    
    out = model(i, s, r)
    print("Output shape:", out.shape) 