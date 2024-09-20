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
    def __init__(self, d_model: int, eps: float):
        '''
            - num_features: number of feats in x-tensor i.e d_model
                - (say) (Batch, seq_len, d_model) 
            - The learnable params "alpha" and "bias" need to interact with each feature in feature-space of input
        '''
        super().__init__()
        # Init to multiplicative id and additive id resp.
        # Only learnable params
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self, x):
        # keepdims = True, for compatibilty during (x - mu) else broadcasting issues
        # Keep the dimension for broadcasting
        mu = x.mean(dim = -1, keepdims=True) # across the feature-space of each seq i.e last dim
        std = x.std(dim = -1, keepdims=True)  # across the feature-space of each seq i.e last dim  

        # Apply xj_ = (xj - mu)/ sqrt( std**2 + eps )
        return (x - mu) / torch.sqrt( std**2 + self.eps  )


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.h = h
        self.dropout = dropout
        self.d_model = d_model

        # Splitting across d_model, so that each head sees the entire seq, but a certain aspect
        # h should divide d_model => Why? 
        assert self.d_model % self.h == 0, "h doesn't divide d_model"
        self.d_k = self.d_model // h

        # Learnable weight tensors that interact with copies of input (i.e k, v, q)
        # Recall in the diag where am I?
        # - (k, v, q) -> (w_k, w_v, w_q)

        # query: (Batch, seq_len, d_model) -> after interaction with w_q -> (Batch, seq_len, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # out weight tensor
        # interacts after concat of all heads
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod # Trap: no"self", need to do this, as we might need to call this w/o am obj i.e instantiating this class
    def attention(key, value, query, mask, dropout):
        '''
        - Can be used as normal MH-A / masked MH-A
        - Returns final o/p after MH-A along with intermediate attention_scores/sim-matrix, for viz
        '''
        
        d_k = query.shape[-1]

        # Imagine for each head, similarity of a token with others .e attention scores
        # (Batch, h, seq_len, d_k) x (Batch, h, d_k, seq_len) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            MinusInf = -1e9
            # masking padded tokens <Used for standardising len> OR
            # masking future tokens in the gt shown to decoder, next token prediction task
            attention_scores = attention_scores.masked_fill_(mask == 0, MinusInf)

        # softmax
        attention_scores = attention_scores.softmax(dim=-1) # max for a given token across that seq

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (Batch, h, seq_len, seq_len) x (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, d_k) 
        out = attention_scores @ value
        return out, attention_scores


    def forward(self, k, q, v, mask, dropout):
        '''
        Mask makes it reuse as both a masked MH-A / MH-A
        '''
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        query = self.w_q(q)
        value = self.w_v(v)

        # Where are we in the diagram?
        # Just before splitting
        # We need for each head -> (seq_len, d_k)
        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        key = key.reshape(key.shape[0], key.shape[1], self.h, self.d_k) # d_model = h * d_k 
        key = key.transpose(1, 2)

        value = value.reshape(value.shape[0], value.shape[1], self.h, self.d_k) # d_model = h * d_k 
        value = value.transpose(1, 2)

        query = query.reshape(query.shape[0], query.shape[1], self.h, self.d_k) # d_model = h * d_k 
        query = query.transpose(1, 2)
        

        out, attention_scores = MultiHeadAttentionBlock.attention(key, value, query, mask, dropout)

        # out -> (Batch, h, seq_len, d_k)
        # concat all the heads for each batch i.e [seq_len, d_k], [seq_len, d_k], ... [h of these]
        # rearrange first
        out = out.transpose(-3, -2)
        # aggregate for each sequence, concat(attention1, attention2, ...)
        out.reshape(out.shape[0], out.shape[1], self.h*self.d_k)

        # multply with out matrix
        return self.w_o(out)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        '''
            As per Paper:
                - x = max(0, xW1 + B1) => relu( fc1(x) ), fc1: d_model = 512, d_ff = 2048
                - x = xW2 + B2 => fc2(x) => fc2: d_ff, d_model
        '''
        # Some papers set bias as False, vanilla transformer doesn't
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff, bias=True) # W1, B1, by defualt bias is True in nn.Linear
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model) # W2, B2
        self.relu = torch.relu

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        # Recall: After activation function dropout is applied. Never before.
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class ResidualConnectionBlock(nn.Module):
    '''
     Idea: 
        - (x_in) -> [ Block ] -> (x_out)
        - out = x_in + x_out

    Args: <As per above>
        sublayer: Block's blueprint
        x_in: tensor to the sublayer 

    NOTE: 
        - As per paper after x_out, apply normm then concat
        - Better practice: 1st norm, then feed to sublayer
    '''

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation(d_model, eps = 10**-6)
        self.d_model = d_model


    def forward(self, x_in, sublayer):
        # apply norm first: Idea taken from Llama
        x = self.norm(x_in)
        x = sublayer(x) # callbacks! Abstract the function signature -> can be FFN forward() or MH-A forward()
        # Apply dropout
        x = self.dropout(x)
        # concat
        x_out = x_in + x
        return x_out


class EncoderBlock(nn.Module):
    '''
    - From the lens of Encoder block, in the paper; 
        - There are N of these,
        - What's the bare min sub-blocks or ingredients needed to construct it?
        - <FeedForward>, <ResidualConnection>, <MH-A>
    '''
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, \
                 feed_forward_layer: FeedForwardBlock, \
                 residual_connection: ResidualConnectionBlock,
                 d_model: int,
                 dropout: nn.Dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.feed_forward_layer = feed_forward_layer
        self.self_attention_block = self_attention_block
        self.norm = LayerNormalisation(d_model, eps = 10**-6)
        
        # There are 2 residual connections in EncoderBlock: Recall
        # - (k, q, v)[x_in] -> MH-A -> x_out
        # x_out = dropout( MH-A(norm(x_in)) )
        # x = x_in + x_out 
        #   => residual's forward will therefore have sublayer as MH-A
        #   => The input first interacts with self.norm
        # Similarly, in the other skip-coonection, subklayer = FFN, norm as usual is first interaction

        # num_features for first interaction is k, q, v's num_feats i.e 512
        # For second interaction, After concat in MH-A, we will restore (Batch, seq_len, d_model) after concat of all attention-heads
            # So, input to FFN's norm will be d_model as well
        # Hence, both the residual's will have num_feats for self.norm as d_model as follows
        # self.residual_connection = nn.ModuleList([residual_connection(d_model, dropout) for _ in range(2)])


    def forward(self, x, src_mask):
        '''
            src_mask is needed for MH-A for padding, so as to standardise seq_len
        '''
        # # x = self.residual_connection[0]() # ditch this approach
        # x = self.residual_connection[0]( x, lambda x: self.self_attention_block(x, x, x, src_mask) )
        # x = self.residual_connection[1]( x, self.feed_forward_layer )

        # norm(x_in) -> MH-A(x, x, x, src_mask) = x -> x__ = [x_in + x]
        x_in = x
        x = self.norm(x)
        x = self.self_attention_block(x, x, x, src_mask, self.dropout)
        x = self.dropout(x)
        x_in = x_in + x # concat/skip

        # norm(x_in) -> FFN(x) = x -> [x_in + x]
        x = self.norm(x_in)
        x = self.feed_forward_layer(x)
        x = self.dropout(x)
        x = x_in + x # concat/skip
        return x


class Encoder(nn.Module):
    '''
    - We need N copies of EncoderBlocks
    - Finally, the last EncoderBlock's o/p need to go through add and norm
    '''
    def __init__(self, encoder_block_layers: nn.ModuleList, N: int, d_model: int ):
        super().__init__()
        self.layers = encoder_block_layers # N of the EncoderBlocks
        self.norm = LayerNormalisation(d_model, eps=10**-6)

    def forward(self, x, src_mask):
        # w/o Input Embedding and Pos Encoding
        for layer in self.layers:
            # out of each EncoderBlock will be 
            x = self.layers(x, src_mask) # This is calling forward() of EncoderBlock

        # finally we have a norm block -> This isn't present in the transformer diag
        # We do this for better practice
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, \
                 cross_attention_block: MultiHeadAttentionBlock, \
                 feed_forward_block: FeedForwardBlock,
                 d_model: int, 
                 dropout: float):
        super().__init__()
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.self_attention_block = self_attention_block
        self.norm = LayerNormalisation(d_model = d_model, eps = 10**-6)
        self.dropout = dropout

        # self.block = nn.ModuleList([ResidualConnectionBlock for _ in range(3)])

    def forward(self, decoder_input, encoder_output, src_mask, tgt_mask):
        '''
            Decoder input here, is after positional encoding
        '''
        # Part1
        x = self.norm(decoder_input)
        x = self.self_attention_block(x, x, x, tgt_mask, self.dropout)
        x = x + decoder_input

        # Part2
        x_in = x
        x = self.norm(x)
        x = self.cross_attention_block(k=encoder_output, v=encoder_output, q=x, mask=src_mask, dropout=self.dropout )
        x = x_in + x

        # Part3
        x_in = x
        x = self.feed_forward_block(x)
        return x + x_in

class Decoder(nn.Module):
    '''
        - Will take all the N blocks of decoder
    '''
    def __init__(self, layers: nn.ModuleList, N:int, d_model: int):
        super().__init__()
        self.layers = layers
        self.N = N
        self.norm = LayerNormalisation(d_model, eps=10**-6)

    def forward(self, decoder_input, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(decoder_input, encoder_output, src_mask, tgt_mask) # layer() is forward() of DecoderBlock 
        return self.norm(x)



class ProjectionLayer(nn.Module):
    '''
        - Projects (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
    '''
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
            Project each item in seq from d_model to vocab_size
            Then take softmax(or log_sm for numerical stabilty) on last dim
        '''
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size) 
        x = self.proj(x)
        return torch.log_softmax(x, dim=-1)


class Transformer(nn.Module):
    '''
     - Structurally, keep encoder(), decoder() and projection() separate
        - Decision to keep encode(), decode(), projection() was made rather than making forward() method for transformer
        - as we might need intermediate attention mask to visualise
        - Also, in decoder, we needn't recompute encode again, rather reuse ther output from encoder()
        - src: meaning source language to translate [Generic task call this as encoder_mask]
        - tgt: Target language to which we translate [Generic: decoder_mask]
    '''
    def __init__(self, \
                src_embed: InputEmbedding, \
                src_pos: PositionalEncoding,
                tgt_embed: InputEmbedding, \
                tgt_pos: PositionalEncoding, \
                encoder: Encoder, \
                decoder: Decoder, \
                projection_layer: ProjectionLayer):
        super().__init__()
        self.src_emb = src_embed
        self.tgt_emb = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        # InputEmbedding -> Positional Enc -> Goes through N Encoder Blocks -> Encoder 
        x = self.src_embed(x)
        x = self.src_pos(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, x, encoder_output, src_mask, tgt_mask):
        # InputEmbedding -> Positional Encoding -> N layers of Decoder Blocks(Decoder) -> Projection 
        x = self.tgt_embed(x)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask) # (decoder_input, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        '''
        - (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        - Applies log_softmax() for next token prediction
        '''
        return self.projection_layer(x)
    



def build_transformer( 
                       src_seq_len: int,
                       tgt_seq_len: int, # Context-window for src and tgt mayn't be same
                       src_vocab_size: int, 
                       tgt_vocab_size: int, # Vocab-size of 2 different languages mostly wont be same
                       d_model: int = 512, \
                       h: int = 8, 
                       d_ff: int = 2048, 
                       N: int = 6,
                       dropout: float = 0.1, 
                      ) -> Transformer: 
    '''
        - Inits all the objects with the CONSTANTS and HYPERPARAMS in the model
        - Uses the hyperparams
    '''

    # InputEmbeddings
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Positional Encodings
    src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

    # create N encoder blocks
    encoder_layers = nn.ModuleList([]) # input for the Encoder
    for _ in range(N):
        self_attention_block_encoder = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        residual_connection = None
        encoder_block = EncoderBlock( 
                                      self_attention_block_encoder, \
                                      encoder_feed_forward, \
                                      residual_connection, \
                                      d_model, 
                                      dropout
                                    )
        encoder_layers.append(encoder_block)

    decoder_layers = nn.ModuleList([]) # input for the Decoder
    for _ in range(N):
        self_attention_block_decoder = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block_decoder = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
                                      self_attention_block_decoder, \
                                      cross_attention_block_decoder,
                                      decoder_feed_forward, 
                                      d_model,
                                      dropout
                                    )
        decoder_layers.append(decoder_block)

    # inputs for Encoder_forward() and Decoder_forward() is prepared
    encoder = Encoder(encoder_layers, N, d_model)
    decoder = Decoder(decoder_layers, N, d_model)
    projection = ProjectionLayer(tgt_vocab_size, d_model)

    # init th Transformer
    transformer = Transformer(src_embed, src_pos, tgt_embed, tgt_pos, encoder, decoder, projection)

    # init all the params of the transformer
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer