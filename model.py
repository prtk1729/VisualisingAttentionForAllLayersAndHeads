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
        # This is a learnable vector, that has 1-1 correpondence with a num-emb
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, \
                                      embedding_dim=self.d_model)

    def forward(self, x):
        '''
            Based on Sec 3.4 of Paper
        '''
        return self.embedding(x) * math.sqrt(self.d_model)




class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # To not overfit
        self.dropout = nn.Dropout(dropout)

        # Skeleton of the PE Matrix
        self.pe = torch.zeros(seq_len, d_model)

        # i: d_model iterator, pos: seq_len iterator
        # theta: pos/ ( 10k**(2i/d_model) )
        # Numerical stabilty Argument: store as x = e^(logx)
        # pe(pos, 2i) = sin(theta), pe(pos, 2i+1) = cos(theta)

        position = torch.arange(0, seq_len, dtype=float).unsqeeze(1)
        two_i = torch.arange(0, d_model, 2, dtype=float) # [0, 2, 4, 6, ..]
        div_term = torch.exp( (two_i/self.d_model) * -math.log(10000.0) )
        theta = position * div_term

        # populate pe
        self.pe[:, 0::2] = torch.sin(theta)
        self.pe[:, 1::2] = torch.cos(theta)

        # we now have for a given seq, but it will come in batches
        self.pe = self.pe.unsqueeze( dim = 0 ) # [ [seq1], [seq2], ...  ]

        # But pe shouldn't be learnable, need to be saved NOT as a learnable param
        # for quick lookup
        self.register_buffer('pe', self.pe) # In the buffer: ["pe": pe_tensor ]



    def forward(self, x):
        ''' 
            Concat "x": Batch of Word-Embs, with corresponsding pe
        '''
        seq_len = x.shape[1] # Need to control this as window
        x_cat = x + self.pe[: , :seq_len, :].requires_grad(False) # Don't track pe, non-learnable
        return self.dropout(x_cat) 



class LayerNormalisation(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        
        # Init to multiplicative id and additive id resp.
        # Only learnable params
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        # keepdims = True, for compatibilty during (x - mu) else broadcasting issues
        mu = x.mean(dim = -1, keepdims=True) # across the feature-space of each seq i.e last dim
        std = x.std(dim = -1, keepdims=True)  # across the feature-space of each seq i.e last dim  

        return (x - mu) / torch.sqrt( std**2 + self.eps  )