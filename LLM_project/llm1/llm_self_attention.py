import torch
import torch.nn as nn


class UstaSelfAttention(nn.Module):
  def __init__(self, embedding_dim, output_dim):
    super().__init__()
    self.embedding_dim = embedding_dim

    self.q_weights = nn.Linear(embedding_dim, output_dim, bias=False)
    self.k_weights = nn.Linear(embedding_dim, output_dim, bias=False)
    self.v_weights = nn.Linear(embedding_dim, output_dim, bias=False)

  def forward(self, x):
    q = self.q_weights(x)
    k = self.k_weights(x)
    v = self.v_weights(x)

    attention_scores = q @ k.T
    attention_weights = torch.softmax(attention_scores / k.shape[-1] ** 0.5, dim=1)
    return attention_weights @ v