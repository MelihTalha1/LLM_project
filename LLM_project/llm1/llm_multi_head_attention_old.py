import torch
import torch.nn as nn

from llm1.llm_causal_attention import UstaCausalAttention


class UstaMultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, output_dim, context_length, num_heads, dropout_rate = 0):
    super().__init__()
    
    self.heads = nn.ModuleList(
      [UstaCausalAttention(embedding_dim, output_dim, context_length, dropout_rate) for _ in range(num_heads)]
    )

    self.projection = nn.Linear(embedding_dim, output_dim)

  def forward(self, x):
    attention_outs = []
    for head in self.heads:
      head_out = head(x)
      attention_outs.append(head_out)

    attention_out = torch.cat(attention_outs, dim=1)

    return self.projection(attention_out)