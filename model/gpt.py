import torch
import torch.nn as nn


def get_sinusoidal_encoding(seq_len, d_model, device):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    i = torch.arange(d_model, dtype=torch.float, device=device).unsqueeze(0)

    angle_rates = 1 / torch.pow(10000, 2*(i//2)/d_model)
    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

    return angle_rads.unsqueeze(0)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_ff, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x_t = x_norm.transpose(0, 1)  # (T, B, D)

        attn_out, _ = self.attn(x_t, x_t, x_t, attn_mask=self._causal_mask(x.size(1), x.device))
        attn_out = attn_out.transpose(0, 1)  # (B, T, D)

        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x

    def _causal_mask(self, size, device):
        mask = torch.tril(torch.ones(size, size, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerDecoderBlock(d_ff, d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        token_embeddings = self.token_embed(x)
        pos_enc = get_sinusoidal_encoding(T, token_embeddings.size(-1), x.device)
        x = token_embeddings + pos_enc
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits