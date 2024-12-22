"""
[Paper]
Author: Yehjin Shin et al. 
Title: "An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention"
Conference: AAAI 2024

[Code Reference]
https://github.com/jeongwhanchoi/BSARec
"""

import torch
import torch.nn as nn

class FrequencyLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, 2:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate, alpha):
        super(TransformerBlock, self).__init__()
        self.alpha = alpha
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_rate)
        self.filter_layer = FrequencyLayer(hidden_size, dropout_rate)
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
        gsp = self.filter_layer(x)
        hidden_states = self.alpha * attn_output + (1-self.alpha) * gsp
        x = self.norm1(x + self.dropout(hidden_states))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class BSARec(nn.Module):
    def __init__(self, args):
        super(BSARec, self).__init__()
        self.num_items = args.num_items + 1
        self.max_seq_length = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.dropout_rate = args.dropout_rate
        self.attn_dropout_rate = args.attn_dropout_rate
        self.num_heads = args.n_heads
        self.num_blocks = args.e_layers
        self.alpha = args.alpha

        self.item_emb = nn.Embedding(self.num_items, self.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, self.num_heads, self.attn_dropout_rate, self.alpha)
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
