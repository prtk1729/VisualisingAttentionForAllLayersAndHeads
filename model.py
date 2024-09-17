import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        '''
        Maps a token-id to a learnable vector of fixed size
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Build a learnable dict kinda that you give a num[token-id] -> it gives a vector
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, \
                                      embedding_dim=self.d_model)

    def forward(self, x):
        '''
            Based on Sec 3.4 of Paper
        '''
        return self.embedding(x) * math.sqrt(self.d_model)


