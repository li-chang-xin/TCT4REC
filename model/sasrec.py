"""
[Paper]
Author: Wang-Cheng Kang et al. 
Title: "Self-Attentive Sequential Recommendation."
Conference: ICDM 2018

[Code Reference]
https://github.com/kang205/SASRec
"""

import torch
import torch.nn as nn

    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class SASRec(nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.num_items = args.num_items + 1
        self.max_seq_length = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.attn_dropout_rate = args.attn_dropout_rate
        self.num_heads = args.n_heads
        self.num_blocks = args.e_layers

        self.item_emb = nn.Embedding(self.num_items, self.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.attn_dropout_rate)
            for _ in range(self.num_blocks)
        ])

        # Final prediction layer
        self.item_bias = nn.Parameter(torch.zeros(self.num_items))

    def forward(self, user_ids, user_discrete_data, input_seq, seq_item_discrete_data,\
                pos_seq, pos_item_discrete_data,\
                    neg_seq, neg_item_discrete_data,\
                     ratings, timestamps):
        seq_emb = self.item_emb(input_seq) + self.pos_emb(torch.arange(self.max_seq_length, device=input_seq.device)).unsqueeze(0)
        seq_emb = self.dropout(seq_emb)

        mask = (input_seq != 0).to(input_seq.device).unsqueeze(-1)  # Shape: [batch_size, max_len, 1]
        seq_emb = seq_emb * mask

        for block in self.attention_blocks:
            seq_emb = block(seq_emb)
            seq_emb = seq_emb * mask
        
        pos_emb = self.item_emb(pos_seq.squeeze(-1))
        neg_emb = self.item_emb(neg_seq.squeeze(-1))
        pos_logits = (seq_emb * pos_emb).sum(dim=-1) + self.item_bias[pos_seq].squeeze()
        neg_logits = (seq_emb * neg_emb).sum(dim=-1) + self.item_bias[neg_seq].squeeze()

        mask = mask.squeeze(-1)
        pos_loss = -torch.log(torch.sigmoid(pos_logits) + 1e-24) * mask
        neg_loss = -torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * mask
        total_loss = torch.sum(pos_loss + neg_loss) / (torch.sum(mask) + 1e-24)

        return total_loss
        
    def predict(self, user_ids, user_discrete_data, input_seq, seq_item_discrete_data,\
                test_items, test_item_discrete_data,  ratings, timestamps):
        
        seq_emb = self.item_emb(input_seq) + self.pos_emb(torch.arange(self.max_seq_length, device=input_seq.device)).unsqueeze(0)
        seq_emb = self.dropout(seq_emb)

        for block in self.attention_blocks:
            seq_emb = block(seq_emb)

        seq_emb = seq_emb[:, -1, :]
        test_emb = self.item_emb(test_items)

        seq_emb_last = seq_emb.unsqueeze(1).expand(-1, test_emb.size(1), -1)

        test_logits = torch.sum(seq_emb_last * test_emb, dim=-1) + self.item_bias[test_items].squeeze()

        return test_logits[:, -1]
