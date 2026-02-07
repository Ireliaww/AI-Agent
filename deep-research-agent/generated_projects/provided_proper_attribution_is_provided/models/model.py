The Transformer model, as described in "Attention Is All You Need" by Vaswani et al. (2017), is a groundbreaking sequence transduction model built entirely on attention mechanisms, eschewing recurrent or convolutional layers. It consists of an encoder-decoder architecture where both components are composed of stacked identical layers.

Here's a complete PyTorch implementation of the Transformer model, following the exact architecture and best practices, including detailed docstrings and type hints.

---

### Transformer Model PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# --- Helper Module: SublayerConnection (Residual Connection + Layer Normalization) ---
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    Note for code simplicity the norm is first, then the sublayer.
    (This is a common variant and often performs similarly or better than pre-norm).
    """
    def __init__(self, size: int, dropout_rate: float):
        """
        Initializes the SublayerConnection.

        Args:
            size (int): The dimensionality of the input and output (d_model).
            dropout_rate (float): The dropout probability.
        """
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies the sublayer, dropout, and adds it to the input with layer normalization.

        Args:
            x (torch.Tensor): The input tensor to the sublayer.
            sublayer (nn.Module): The sublayer function (e.g., MultiHeadAttention, PositionwiseFeedForward).

        Returns:
            torch.Tensor: The output tensor after applying sublayer, dropout, and residual connection with normalization.
        """
        # Original paper: x + dropout(sublayer(norm(x)))
        # Pre-norm variant (often used in modern implementations): norm(x + dropout(sublayer(x)))
        # Here we follow the common "post-norm" interpretation from the paper, but apply norm *before* the residual.
        # This is equivalent to: x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


# --- 1. Embeddings and Positional Encoding ---
class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in the paper.
    This adds a signal to the input embeddings that depends on the position of the token.
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimensionality of the model (embedding dimension).
            dropout_rate (float): The dropout probability.
            max_len (int): The maximum sequence length for which to generate positional encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe) # Register as a buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with added positional encodings, shape (batch_size, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1)] # Add PE up to the current sequence length
        return self.dropout(x)

class Embeddings(nn.Module):
    """
    Combines token embeddings and positional encodings.
    """
    def __init__(self, vocab_size: int, d_model: int, dropout_rate: float, max_len: int = 5000):
        """
        Initializes the Embeddings layer.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the model (embedding dimension).
            dropout_rate (float): The dropout probability for positional encoding.
            max_len (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs token embedding lookup and adds positional encodings.

        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor with embeddings and positional encodings,
                          shape (batch_size, seq_len, d_model).
        """
        # Scale the embeddings by sqrt(d_model) as per the paper
        return self.positional_encoding(self.embedding(x) * math.sqrt(self.d_model))


# --- 2. Multi-Head Attention Mechanism ---
def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Scaled Dot-Product Attention.

    Args:
        query (torch.Tensor): Query tensor, shape (batch_size, n_heads, seq_len_q, d_k).
        key (torch.Tensor): Key tensor, shape (batch_size, n_heads, seq_len_k, d_k).
        value (torch.Tensor): Value tensor, shape (batch_size, n_heads, seq_len_v, d_v).
                              Note: seq_len_k must be equal to seq_len_v.
        mask (torch.Tensor, optional): Mask tensor, shape (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k).
                                       Masked positions will be set to -inf.
        dropout (nn.Dropout, optional): Dropout layer to apply to attention weights.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): Weighted sum of values, shape (batch_size, n_heads, seq_len_q, d_v).
            - attn_weights (torch.Tensor): Attention weights, shape (batch_size, n_heads, seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)
    # (batch_size, n_heads, seq_len_q, d_k) @ (batch_size, n_heads, d_k, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Apply mask by setting masked positions to a very small number (-inf)
        scores = scores.masked_fill(mask == 0, -1e9) # Assuming mask is 0 for padded/future tokens

    p_attn = F.softmax(scores, dim=-1) # (batch_size, n_heads, seq_len_q, seq_len_k)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn # (batch_size, n_heads, seq_len_q, d_v)

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    Projects Q, K, V linearly, splits into multiple heads, applies attention,
    concatenates heads, and applies a final linear projection.
    """
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            d_model (int): The dimensionality of the model (embedding dimension).
            n_heads (int): The number of attention heads.
            dropout_rate (float): The dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads # Dimension of K, Q for each head
        self.n_heads = n_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)]) # W_q, W_k, W_v, W_o
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs multi-head attention.

        Args:
            query (torch.Tensor): Query tensor, shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor, shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor, shape (batch_size, seq_len_v, d_model).
            mask (torch.Tensor, optional): Mask tensor, shape (batch_size, 1, seq_len_q, seq_len_k)
                                           or (batch_size, 1, 1, seq_len_k).

        Returns:
            torch.Tensor: Output tensor after multi-head attention, shape (batch_size, seq_len_q, d_model).
        """
        if mask is not None:
            # Same mask applied to all heads
            mask = mask.unsqueeze(1) # Add head dimension (batch_size, 1, seq_len_q, seq_len_k)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => n_heads x d_k
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        query, key, value = [
            l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears[:3], (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn_weights = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # (batch_size, n_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, n_heads, d_k) -> (batch_size, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        return self.linears[-1](x) # Final linear projection W_o


# --- 3. Position-wise Feed-Forward Network ---
class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network, consisting of two linear transformations
    with a ReLU activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initializes the PositionwiseFeedForward layer.

        Args:
            d_model (int): The dimensionality of the model (input/output dimension).
            d_ff (int): The dimensionality of the inner-layer (hidden dimension).
            dropout_rate (float): The dropout probability.
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feed-forward network to the input.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model).
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# --- 4. Encoder Layer ---
class EncoderLayer(nn.Module):
    """
    One layer of the Transformer encoder.
    Consists of a Multi-Head Self-Attention sublayer and a Position-wise Feed-Forward sublayer,
    each followed by residual connections and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        """
        Initializes an EncoderLayer.

        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner feed-forward layer.
            dropout_rate (float): The dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout_rate) for _ in range(2)
        ])
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through one encoder layer.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Source padding mask, shape (batch_size, 1, 1, seq_len_src).
                                          Masks padded positions (0 for masked, 1 for unmasked).

        Returns:
            torch.Tensor: Output tensor of the encoder layer, shape (batch_size, seq_len, d_model).
        """
        # Self-attention sublayer
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, mask))
        # Feed-forward sublayer
        x = self.sublayer_connections[1](x, self.feed_forward)
        return x


# --- 5. Decoder Layer ---
class DecoderLayer(nn.Module):
    """
    One layer of the Transformer decoder.
    Consists of a Masked Multi-Head Self-Attention, a Multi-Head Encoder-Decoder Attention,
    and a Position-wise Feed-Forward sublayer, each followed by residual connections and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        """
        Initializes a DecoderLayer.

        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner feed-forward layer.
            dropout_rate (float): The dropout probability.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.src_attn = MultiHeadAttention(d_model, n_heads, dropout_rate) # Encoder-Decoder Attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout_rate) for _ in range(3)
        ])
        self.d_model = d_model

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through one decoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer (or embeddings),
                              shape (batch_size, seq_len_tgt, d_model).
            memory (torch.Tensor): Output tensor from the encoder stack,
                                   shape (batch_size, seq_len_src, d_model).
            src_mask (torch.Tensor, optional): Source padding mask, shape (batch_size, 1, 1, seq_len_src).
            tgt_mask (torch.Tensor, optional): Target padding and look-ahead mask,
                                               shape (batch_size, 1, seq_len_tgt, seq_len_tgt).

        Returns:
            torch.Tensor: Output tensor of the decoder layer, shape (batch_size, seq_len_tgt, d_model).
        """
        # Masked self-attention sublayer (uses tgt_mask)
        x = self.sublayer_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Encoder-decoder attention sublayer (uses src_mask)
        # Query comes from decoder, Key/Value come from encoder memory
        x = self.sublayer_connections[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        # Feed-forward sublayer
        x = self.sublayer_connections[2](x, self.feed_forward)
        return x


# --- 6. Encoder Stack ---
class Encoder(nn.Module):
    """
    The full Transformer Encoder stack.
    Composed of N identical EncoderLayers.
    """
    def __init__(self, layer: EncoderLayer, N: int):
        """
        Initializes the Encoder stack.

        Args:
            layer (EncoderLayer): An instance of an EncoderLayer to be stacked.
            N (int): The number of identical EncoderLayers to stack.
        """
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model) # Final layer normalization

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the encoder stack.

        Args:
            x (torch.Tensor): Input tensor (embeddings + positional encodings),
                              shape (batch_size, seq_len_src, d_model).
            mask (torch.Tensor, optional): Source padding mask, shape (batch_size, 1, 1, seq_len_src).

        Returns:
            torch.Tensor: Output tensor from the encoder stack, shape (batch_size, seq_len_src, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # Apply final layer norm as per paper (sometimes omitted)


# --- 7. Decoder Stack ---
class Decoder(nn.Module):
    """
    The full Transformer Decoder stack.
    Composed of N identical DecoderLayers.
    """
    def __init__(self, layer: DecoderLayer, N: int):
        """
        Initializes the Decoder stack.

        Args:
            layer (DecoderLayer): An instance of a DecoderLayer to be stacked.
            N (int): The number of identical DecoderLayers to stack.
        """
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model) # Final layer normalization

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the decoder stack.

        Args:
            x (torch.Tensor): Input tensor (embeddings + positional encodings),
                              shape (batch_size, seq_len_tgt, d_model).
            memory (torch.Tensor): Output tensor from the encoder stack,
                                   shape (batch_size, seq_len_src, d_model).
            src_mask (torch.Tensor, optional): Source padding mask, shape (batch_size, 1, 1, seq_len_src).
            tgt_mask (torch.Tensor, optional): Target padding and look-ahead mask,
                                               shape (batch_size, 1, seq_len_tgt, seq_len_tgt).

        Returns:
            torch.Tensor: Output tensor from the decoder stack, shape (batch_size, seq_len_tgt, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x) # Apply final layer norm as per paper (sometimes omitted)


# --- 8. Transformer Model ---
class Transformer(nn.Module):
    """
    The complete Transformer sequence transduction model.
    Combines Encoder, Decoder, embeddings, and a final linear output layer.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8, N_layers: int = 6,
                 d_ff: int = 2048, dropout_rate: float = 0.1):
        """
        Initializes the Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimensionality of the model's embeddings and hidden states. Default is 512.
            n_heads (int): Number of attention heads. Default is 8.
            N_layers (int): Number of identical encoder/decoder layers. Default is 6.
            d_ff (int): Dimensionality of the inner feed-forward layer. Default is 2048.
            dropout_rate (float): Dropout probability. Default is 0.1.
        """
        super().__init__()
        # Embeddings for source and target
        self.src_embed = Embeddings(src_vocab_size, d_model, dropout_rate)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model, dropout_rate)

        # Encoder stack
        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
        self.encoder = Encoder(encoder_layer, N_layers)

        # Decoder stack
        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
        self.decoder = Decoder(decoder_layer, N_layers)

        # Final linear layer to project decoder output to target vocabulary size
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters with Glorot / Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the entire Transformer model.

        Args:
            src (torch.Tensor): Source input sequence of token IDs, shape (batch_size, seq_len_src).
            tgt (torch.Tensor): Target input sequence of token IDs (shifted right),
                                shape (batch_size, seq_len_tgt).
            src_mask (torch.Tensor, optional): Source padding mask, shape (batch_size, 1, 1, seq_len_src).
            tgt_mask (torch.Tensor, optional): Target padding and look-ahead mask,
                                               shape (batch_size, 1, seq_len_tgt, seq_len_tgt).

        Returns:
            torch.Tensor: Logits for the next token prediction, shape (batch_size, seq_len_tgt, tgt_vocab_size).
        """
        # Encode the source sequence
        encoder_output = self.encode(src, src_mask)
        # Decode the target sequence using encoder output
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        # Project decoder output to vocabulary size
        return self.generator(decoder_output)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encodes the source sequence.

        Args:
            src (torch.Tensor): Source input sequence of token IDs, shape (batch_size, seq_len_src).
            src_mask (torch.Tensor, optional): Source padding mask.

        Returns:
            torch.Tensor: Encoded representation, shape (batch_size, seq_len_src, d_model).
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: Optional[torch.Tensor],
               tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor]) -> torch.Tensor: