import math

import torch
from torch import nn
from torch.nn import functional as F


class SASRec(nn.Module):
    def __init__(self, num_items, max_his_len=20, hidden_dim=32, num_layers=2,
                 num_heads=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_his_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, eps=1e-8) for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim, eps=1e-8) for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
            ) for _ in range(num_layers)
        ])
        self.last_norm = nn.LayerNorm(hidden_dim, eps=1e-8)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, his):
        valid = his.ne(0).unsqueeze(-1)
        positions = torch.arange(his.size(1), device=his.device)
        states = self.item_embedding(his) * math.sqrt(self.hidden_dim)
        states = self.dropout(states + self.position_embedding(positions)) * valid
        causal_mask = torch.ones(his.size(1), his.size(1), device=his.device,
                                 dtype=torch.bool).triu(1)
        for norm, attention, ffn_norm, ffn in zip(
            self.attention_norms, self.attention_layers, self.ffn_norms, self.ffn_layers
        ):
            # Kang and McAuley (2018), III-B/C: causal attention and point-wise FFN.
            query = norm(states)
            update, _ = attention(query, states, states, attn_mask=causal_mask,
                                  key_padding_mask=~his.ne(0), need_weights=False)
            states = query + update
            normalized = ffn_norm(states)
            states = (normalized + ffn(normalized)) * valid
        return self.last_norm(states) * valid

    def loss(self, his, pos, neg):
        states = self(his)
        valid = pos.ne(0)
        positive = (states * self.item_embedding(pos)).sum(dim=-1)[valid]
        negative = (states * self.item_embedding(neg)).sum(dim=-1)[valid]
        return F.softplus(-positive).mean() + F.softplus(negative).mean()

    def predict(self, his):
        states = self(his)
        last = his.ne(0).sum(dim=1) - 1
        users = states[torch.arange(len(his), device=his.device), last]
        return users @ self.item_embedding.weight[1:].t()
