import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class PositionalEncoding(nn.Module):#######################################
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        r = self.pe[:, :x.size(1)].expand_as(x)
        x = x + r
        return self.dropout(x)################################

class MultiHeadAttention(nn.Module):################################
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        N = q.size(0)  # Batch size
        L = q.size(1)  # Sequence length
        Q = self.q_linear(q).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(N, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = Q.matmul(K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        
        out = attention.matmul(V).transpose(1, 2).contiguous().view(N, -1, self.num_heads * self.d_k)
        return self.fc_out(out)################################

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha =   nn.MultiheadAttention(d_model, num_heads, dropout)#################
        self.ivert_mha = nn.MultiheadAttention(d_model, num_heads, dropout)
        
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ivert_ff = FeedForward(d_model, d_ff, dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ivert_layernorm1 = nn.LayerNorm(d_model)
        self.ivert_layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, attn_mask = None):
        attn_output, _ = self.mha(x, x, x, attn_mask)
        invet_attn_output, _ = self.ivert_mha(y, y, y)##################
        x = self.layernorm1(x + self.dropout(attn_output))
        y = self.ivert_layernorm1(y + self.dropout(invet_attn_output))
        ff_x_output = self.ff(x)
        ff_y_output = self.ivert_ff(y)
        return self.layernorm2(x + self.dropout(ff_x_output)), self.ivert_layernorm2(y + self.dropout(ff_y_output))

class Encoder(nn.Module):
    def __init__(self, input_dim, invert_input_dim, static_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model, padding_idx=0)##长度不变
        self.ivert_embedding = nn.Linear(invert_input_dim, d_model)
        self.static_embedding = nn.Linear(static_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.projector = nn.Linear(d_model, invert_input_dim, bias=True)
        #self.positional_encoding = PositionalEncoding(d_model, dropout)################################################
        self.pos_emb = nn.Embedding(invert_input_dim, d_model)
        self.max_seq_length = invert_input_dim
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, x_mark=None, static_covariates=None, mask = None):
        x = self.embedding(x) + self.pos_emb(torch.arange(self.max_seq_length, device=x.device)).unsqueeze(0)################################
        x = self.dropout(x)################################
        #x = self.positional_encoding(x)################################
        _, _, N = x_mark.shape

        if x_mark is not None:
            y = self.ivert_embedding(x_mark.permute(0, 2, 1))
            #y = self.ivert_embedding(torch.cat([y, x_mark.permute(0, 2, 1)], 1))

        if static_covariates is not None:
            static_embed = self.static_embedding(static_covariates.to(torch.float))
            y = y + static_embed.unsqueeze(1)

        y = self.dropout(y)

        if mask is not None:
            x = x * mask
            
        for layer in self.layers:
            x, y= layer(x, y)
            if mask is not None:
                x = x * mask

        y = self.projector(y).permute(0, 2, 1)[:, :, : N - 2]

        return x, y

class Bidimensional_Transformer(nn.Module):
    def __init__(self, input_dim, invert_input_dim, static_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Bidimensional_Transformer, self).__init__()
        self.encoder = Encoder(input_dim, invert_input_dim, static_size, d_model, num_heads, d_ff, num_layers, dropout)

    def forward(self, x, x_mark, static_covariates=None, mask=None):
        output, invert_output = self.encoder(x, x_mark, static_covariates, mask)
        return output, invert_output
