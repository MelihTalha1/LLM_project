import torch
import torch.nn as nn


class UstaMultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate = 0, device="cpu"):
    super().__init__()

    self.context_length = context_length
    
    self.multi_head_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, device=device)
    self.projection = nn.Linear(embedding_dim, output_dim, device=device)

    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool().to(device))

  def forward(self, x):
    number_of_tokens = x.shape[0]
    x = x[:self.context_length]
    attention_mask = self.mask[:number_of_tokens, :number_of_tokens]
    out, _ = self.multi_head_attention(x, x, x, attn_mask=attention_mask)
    out = self.projection(out)
    return out