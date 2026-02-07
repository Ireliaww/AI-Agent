The Transformer model, as described in the methodology, is a groundbreaking sequence transduction model that eschews recurrence and convolutions entirely, relying instead on multi-headed self-attention mechanisms. This architecture allows for parallel computation across input sequences, significantly improving training efficiency and enabling the modeling of long-range dependencies more effectively.

Below is a complete PyTorch implementation of the Transformer model, adhering to the specified requirements.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, List, Tuple

# --- Helper Functions and Modules ---

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Produce N identical layers.
    Used for stacking multiple encoder or decoder layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """
    A LayerNormalization module as introduced in the paper.
    Applies normalization over the last dimension.
    """
    def __init__(self, features: int, eps: float = 1e-6):
        """
        Initializes the LayerNorm module.

        Args:
            features (int): The number of features (dimension of the input).
            eps (float): A small epsilon value to prevent division by zero.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs layer normalization on the input tensor.

        Args:
            x (torch.Tensor): Input tensor, typically of shape (batch_size, seq_len, features).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for simplicity the norm is first to avoid a residual connection
    around the first sublayer in the decoder.
    """
    def __init__(self, size: int, dropout: float):
        """
        Initializes the SublayerConnection.

        Args:
            size (int): The dimension of the model (d_model).
            dropout (float): Dropout rate.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies a residual connection followed by layer normalization.

        Args:
            x (torch.Tensor): The input to the sublayer.
            sublayer (nn.Module): The sublayer module (e.g., attention or FFN).

        Returns:
            torch.Tensor: The output of the sublayer, normalized and with residual connection.
        """
        # "Add & Norm" in the paper is actually "Norm then Add" for pre-norm Transformers.
        # The original paper used post-norm, but pre-norm often performs better.
        # For strict adherence to the paper's original "post-norm":
        # return x + self.dropout(sublayer(self.norm(x)))
        # For simplicity and common practice (pre-norm):
        return x + self.dropout(sublayer(self.norm(x)))


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Generate a mask for subsequent positions (look-ahead mask).
    This ensures that each position can only attend to earlier positions.
    Used in the decoder's self-attention to prevent attending to future tokens.

    Args:
        size (int): The sequence length.

    Returns:
        torch.Tensor: A square upper triangular matrix of shape (1, size, size)
                      with 0s on and below the diagonal, and -inf above.
    """
    attn_shape = (1, size, size)
    # `subsequent_mask` is a lower triangular matrix
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0 # Invert to get a mask where 0s allow attention, 1s block (or -inf)


# --- Core Transformer Components ---

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding.
    Adds information about the relative or absolute position of tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model (d_model).
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length to pre-compute positional encodings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension for broadcasting
        self.register_buffer('pe', pe) # Store as buffer, not parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor (embeddings) of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Input tensor with positional encodings added, shape (batch_size, seq_len, d_model).
        """
        # Slice positional encodings to match input sequence length
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, d_k).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, d_k).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, d_v).
                              Note: seq_len_k must be equal to seq_len_v.
        mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, 1, 1, seq_len_k) or
                                       (batch_size, 1, seq_len_q, seq_len_k).
                                       Typically 0s for positions to attend, -inf for masked positions.
        dropout (Optional[nn.Dropout]): Dropout layer to apply to attention weights.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Output tensor of shape (batch_size, num_heads, seq_len_q, d_v).
            - Attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)
    # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Mask out positions with -inf so they don't contribute to softmax
        scores = scores.masked_fill(mask == 0, -1e9) # Use 0 for valid, 1 for masked in the mask tensor

    p_attn = F.softmax(scores, dim=-1) # Apply softmax over the last dimension (seq_len_k)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.
    Combines h independent attention heads.
    """
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        Initializes the MultiHeadAttention module.

        Args:
            h (int): Number of attention heads.
            d_model (int): Dimension of the model (d_model). Must be divisible by h.
            dropout (float): Dropout rate.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by num_heads (h)"
        self.d_k = d_model // h # Dimension of each head's key/query/value
        self.h = h
        # Linear layers for Q, K, V projections and final output projection
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None # Stores attention weights for visualization/debugging
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, 1, seq_len_k) or
                                           (batch_size, seq_len_q, seq_len_k).
                                           Will be broadcasted to (batch_size, num_heads, seq_len_q, seq_len_k).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # Add head dimension for broadcasting

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h*d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    Applied to each position separately and identically.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the PositionwiseFeedForward module.

        Args:
            d_model (int): The dimension of the model (d_model).
            d_ff (int): The dimension of the inner layer (feed-forward dimension).
            dropout (float): Dropout rate.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the feed-forward operation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# --- Encoder ---

class EncoderLayer(nn.Module):
    """
    One layer of the Transformer Encoder.
    Consists of self-attention and a feed-forward network, each followed by
    a residual connection and layer normalization.
    """
    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        """
        Initializes an EncoderLayer.

        Args:
            size (int): The dimension of the model (d_model).
            self_attn (MultiHeadAttention): Self-attention module.
            feed_forward (PositionwiseFeedForward): Position-wise feed-forward module.
            dropout (float): Dropout rate.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # Two sublayers
        self.size = size

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Padding mask for self-attention,
                                           shape (batch_size, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention sublayer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Feed-forward sublayer
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """
    The Transformer Encoder stack.
    Composed of an input embedding, positional encoding, and N identical encoder layers.
    """
    def __init__(self, layer: EncoderLayer, N: int):
        """
        Initializes the Encoder stack.

        Args:
            layer (EncoderLayer): An instance of the EncoderLayer to be cloned.
            N (int): The number of identical encoder layers.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size) # Final layer normalization

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the encoder stack.

        Args:
            x (torch.Tensor): Input tensor (embeddings + positional encodings)
                              of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Padding mask for encoder self-attention,
                                           shape (batch_size, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# --- Decoder ---

class DecoderLayer(nn.Module):
    """
    One layer of the Transformer Decoder.
    Consists of masked self-attention, encoder-decoder attention, and a feed-forward network.
    Each sublayer is followed by a residual connection and layer normalization.
    """
    def __init__(self, size: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward, dropout: float):
        """
        Initializes a DecoderLayer.

        Args:
            size (int): The dimension of the model (d_model).
            self_attn (MultiHeadAttention): Masked self-attention module for decoder input.
            src_attn (MultiHeadAttention): Encoder-decoder attention module (attends to encoder output).
            feed_forward (PositionwiseFeedForward): Position-wise feed-forward module.
            dropout (float): Dropout rate.
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn # Attention over source (encoder output)
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # Three sublayers

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the decoder layer.

        Args:
            x (torch.Tensor): Decoder input (target embeddings + positional encodings)
                              of shape (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output from the encoder stack,
                                   shape (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Padding mask for encoder output (src_mask),
                                                shape (batch_size, 1, src_seq_len).
            tgt_mask (Optional[torch.Tensor]): Combined padding and look-ahead mask for decoder self-attention,
                                                shape (batch_size, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        m = memory # Renaming for clarity

        # 1. Masked self-attention over decoder input
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2. Encoder-decoder attention (query from decoder, key/value from encoder output)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3. Feed-forward network
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """
    The Transformer Decoder stack.
    Composed of N identical decoder layers.
    """
    def __init__(self, layer: DecoderLayer, N: int):
        """
        Initializes the Decoder stack.

        Args:
            layer (DecoderLayer): An instance of the DecoderLayer to be cloned.
            N (int): The number of identical decoder layers.
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size) # Final layer normalization

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the decoder stack.

        Args:
            x (torch.Tensor): Decoder input (target embeddings + positional encodings)
                              of shape (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output from the encoder stack,
                                   shape (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Padding mask for encoder output (src_mask),
                                                shape (batch_size, 1, src_seq_len).
            tgt_mask (Optional[torch.Tensor]): Combined padding and look-ahead mask for decoder self-attention,
                                                shape (batch_size, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# --- Full Transformer Model ---

class Embeddings(nn.Module):
    """
    Simple word embedding layer with a scaling factor.
    """
    def __init__(self, d_model: int, vocab: int):
        """
        Initializes the Embeddings module.

        Args:
            d_model (int): The dimension of the model (d_model).
            vocab (int): The size of the vocabulary.
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model) # Lookup table for embeddings
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs embedding lookup and scales the output.

        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tokens of shape (batch_size, seq_len, d_model).
        """
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    """
    Standard linear + softmax generation step.
    Maps the decoder's output to the vocabulary space.
    """
    def __init__(self, d_model: int, vocab: int):
        """
        Initializes the Generator module.

        Args:
            d_model (int): The dimension of the model (d_model).
            vocab (int): The size of the output vocabulary.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the decoder output to vocabulary logits.

        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Logits for each vocabulary token, shape (batch_size, seq_len, vocab).
        """
        return F.log_softmax(self.proj(x), dim=-1) # Log softmax for numerical stability

class Transformer(nn.Module):
    """
    The complete Transformer model as an Encoder-Decoder architecture.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Embeddings,
                 tgt_embed: Embeddings, generator: Generator):
        """
        Initializes the full Transformer model.

        Args:
            encoder (Encoder): The Encoder stack.
            decoder (Decoder): The Decoder stack.
            src_embed (Embeddings): Source embedding layer.
            tgt_embed (Embeddings): Target embedding layer.
            generator (Generator): Output generation layer (linear + softmax).
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Transformer model.

        Args:
            src (torch.Tensor): Source input token IDs of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target input token IDs of shape (batch_size, tgt_seq_len).
            src_mask (torch.Tensor): Padding mask for source input,
                                     shape (batch_size, 1, src_seq_len).
            tgt_mask (torch.Tensor): Combined padding and look-ahead mask for target input,
                                     shape (batch_size, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Logits for the next token prediction,
                          shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): Source input token IDs.
            src_mask (torch.Tensor): Padding mask for source input.

        Returns:
            torch.Tensor: Encoder output (memory) of shape (batch_size, src_seq_len, d_model).
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the target sequence given encoder output.

        Args:
            memory (torch.Tensor): Output from the encoder stack.
            src_mask (torch.Tensor): Padding mask for encoder output.
            tgt (torch.Tensor): Target input token IDs.
            tgt_mask (torch.Tensor): Combined padding and look-ahead mask for target input.

        Returns:
            torch.Tensor: Decoder output (before final generator layer)
                          of shape (batch_size, tgt_seq_len, d_model).
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# --- Model Instantiation and Utility ---

def make_model(src_vocab: int, tgt_vocab: int, N: int = 6,
               d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1) -> Transformer:
    """
    Helper function to construct a Transformer model from hyperparameters.

    Args:
        src_vocab (int): Size of the source vocabulary.
        tgt_vocab (int): Size of the target vocabulary.
        N (int): Number of encoder and decoder layers.
        d_model (int): Dimension of the model (embedding size).
        d_ff (int): Dimension of the feed-forward inner layer.
        h (int): Number of attention heads.
        dropout (float): Dropout rate.

    Returns:
        Transformer: A complete Transformer model instance.
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = Transformer(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Glorot / Xavier initialization
    # This is important for stable training
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# --- Example Usage and Edge Cases ---

if __name__ == '__main__':
    # 1. Hyperparameters
    src_vocab_size = 1000  # Example source vocabulary size
    tgt_vocab_size = 1000  # Example target vocabulary size
    d_model = 512
    N = 6                # Number of encoder/decoder layers
    h = 8                # Number of attention heads
    dropout = 0.1
    d_ff = 2048          # Feed-forward dimension

    # 2. Instantiate the model
    model = make_model(src_vocab_size, tgt_vocab_size, N, d_model, d_ff, h, dropout)
    print("Transformer Model Architecture:\n", model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. Example input data
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8

    # Random input token IDs
    src_data = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    # Create masks
    # For source (encoder self-attention): padding mask
    # Assume 0 is padding token ID. Here, we'll just create a dummy mask for illustration.
    # A real padding mask would be `(src_data != 0).unsqueeze(1)`
    src_mask = torch.ones(batch_size, 1, src_seq_len).bool() # No padding for simplicity here
    # For target (decoder self-attention): look-ahead mask AND padding mask
    # A real padding mask for target would be `(tgt_data != 0).unsqueeze(1)`