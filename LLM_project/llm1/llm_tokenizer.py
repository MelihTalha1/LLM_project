import json

import torch


class UstaTokenizer:
  def __init__(self, vocab_file):
    with open(vocab_file, "r") as f:
      self.vocab = json.load(f)
      self.reverse_vocab = {v: k for k, v in self.vocab.items()}

  def encode_batch(self, texts, context_length):
    sentences_tokens = []
    for text in texts:
      tokens = self.encode(text).tolist()
      if len(tokens) > context_length:
        tokens = tokens[:context_length]
      else:
        tokens = tokens + [self.vocab["<pad>"]] * (context_length - len(tokens))

      sentences_tokens.append(tokens)

    return torch.tensor(sentences_tokens)
   
  def encode(self, text):
    tokens = [] 
       
    for word in text.split():
      i = 0
      # example: states
      # state => 4
      # s => 58
      while i < len(word):
        found_match = False
        for j in range(len(word), i, -1):
          sub_word = word[i:j]
          if sub_word in self.vocab:
            tokens.append(self.vocab[sub_word])
            i = j
            found_match = True
            break
        if not found_match:
          tokens.append(self.vocab["<unk>"])
          i += 1
      tokens.append(self.vocab[" "])

    # check if text is not ends with a space
    if not text.endswith(" "):
      tokens.pop()
    return torch.tensor(tokens)
  
  def tokenize(self, text):
    token_ids = self.encode(text)
    # token_ids from tensor to list
    token_ids = token_ids.detach().numpy().tolist()

    return [self.reverse_vocab[id] for id in token_ids]

  def decode(self, ids):
    text = ""
    for id in ids:
      text += self.reverse_vocab[id]
    return text