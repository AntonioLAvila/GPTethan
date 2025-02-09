import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model) # query
        self.W_k = nn.Linear(d_model, d_model) # key
        self.W_v = nn.Linear(d_model, d_model) # value
        self.W_o = nn.Linear(d_model, d_model) # output layer

    def attention(self, Q, K, V, mask=None):
        attn = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(self.d_k)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn_prob = torch.softmax(attn, dim=-1)

        return torch.matmul(attn_prob, V)
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K , V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_out = self.attention(Q, K, V, mask)

        return self.W_o(self.combine_heads(attn_out))
    
class FF(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FF, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return \
        self.fc2(
            self.relu(
                self.fc1(x)
            )
        )
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FF(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FF(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, target_mask):
        attn_out = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, target):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3).to(target.device)
        seq_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=target.device), diagonal=1)).bool()
        target_mask = target_mask & nopeak_mask
        return src_mask, target_mask

    def forward(self, src, target):
        src_mask, target_mask = self.generate_mask(src, target)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = target_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, target_mask)

        output = self.fc(dec_output)
        return output