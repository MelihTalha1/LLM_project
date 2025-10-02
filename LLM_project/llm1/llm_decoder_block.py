import torch.nn as nn

from llm1.llm_layer_norm import UstaLayerNorm
from llm1.llm_mlp import UstaMLP
from llm1.llm_multi_head_attention import UstaMultiHeadAttention


class UstaDecoderBlock(nn.Module):
  def __init__(self, embedding_dim, num_heads, context_length, device):
    super().__init__()

    self.self_attention = UstaMultiHeadAttention(
      embedding_dim,
      embedding_dim,
      context_length,
      num_heads,
      dropout_rate=0.5,
      device=device
    )
    self.norm1 = UstaLayerNorm(embedding_dim, device=device)
    self.mlp = UstaMLP(embedding_dim, embedding_dim, device=device)
    self.norm2 = UstaLayerNorm(embedding_dim, device=device)

  def forward(self, x):
    res = self.norm1(x)
    x = self.self_attention(x)
    x = self.norm1(x)

    x = x + res

    res = self.norm2(x)
    x = self.mlp(x)
    x = self.norm2(x)

    x = x + res

    return x