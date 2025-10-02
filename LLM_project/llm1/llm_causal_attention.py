import torch
import torch.nn as nn


class UstaCausalAttention(nn.Module):
  def __init__(self, embedding_dim, output_dim, context_length, dropout_rate = 0):
    super().__init__()
    self.embedding_dim = embedding_dim

    self.q_weights = nn.Linear(embedding_dim, output_dim, bias=False)
    self.k_weights = nn.Linear(embedding_dim, output_dim, bias=False)
    self.v_weights = nn.Linear(embedding_dim, output_dim, bias=False)
    self.dropout = nn.Dropout(dropout_rate)
    self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))
    self.context_length = context_length

  def forward(self, x):
    number_of_tokens = x.shape[0]
    # truncate the context length to the context length of the model
    x = x[:self.context_length]
    q = self.q_weights(x)
    k = self.k_weights(x)
    v = self.v_weights(x)

    attention_scores = q @ k.T
    attention_scores = attention_scores.masked_fill_(
      self.mask.bool()[:number_of_tokens, :number_of_tokens] == 0, -torch.inf
    )
    attention_scores = torch.softmax(attention_scores / k.shape[-1] ** 0.5, dim=1)
    attention_scores = self.dropout(attention_scores)

    return attention_scores @ v