The Transformer model, as introduced in "Attention Is All You Need" by Vaswani et al., revolutionized sequence transduction by relying entirely on attention mechanisms, eschewing recurrent or convolutional layers. This implementation provides a modular PyTorch version of the original Transformer architecture.

---

### PyTorch Implementation of the Transformer Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable

# --- Helper Functions for Masking ---

def generate_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Generates a padding mask for a given sequence.

    Args:
        seq (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
        pad_idx (int): The index of the padding token.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, 1, 1, seq_len)
                      where True indicates a padding token (will be masked out).
    """
    # (batch_size, 1, 1, seq_len) for broadcasting with attention scores
    return (seq == pad_idx).unsqueeze(1).unsqueeze(1)


def generate_look_ahead_mask(seq_len: int) -> torch.Tensor:
    """
    Generates a look-ahead mask to prevent attention to future tokens.

    Args:
        seq_len (int): The length of the sequence.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (seq_len, seq_len)
                      where True indicates positions to be masked out (future tokens).
    """
    # Upper triangular matrix of ones, then convert to boolean
    # e.g., for seq_len=3:
    # [[0., 1., 1.],
    #  [0., 0., 1.],
    #  [0., 0., 0.]]
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return look_ahead_mask


# --- Core Components of the Transformer ---

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding to inject information about the relative or
    absolute position of tokens in the sequence.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings.
        max_len (int): The maximum sequence length expected.
        dropout (nn.Dropout): Dropout layer for regularization.
        pe (torch.Tensor): Precomputed positional encoding matrix.
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encodings added,
                          shape (batch_size, seq_len, d_model).
        """
        # Add positional encodings to the input embeddings
        # x has shape (batch_size, seq_len, d_model)
        # self.pe has shape (1, max_len, d_model)
        # We slice self.pe up to the current sequence length.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None) -> torch.Tensor:
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k).
        key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k).
        value (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v).
                              Note: seq_len_k must be equal to seq_len_v.
        mask (Optional[torch.Tensor]): Mask tensor of shape (..., seq_len_q, seq_len_k).
                                       Positions with True will be masked out (set to -inf).
        dropout (Optional[nn.Dropout]): Dropout layer to apply to attention weights.

    Returns:
        torch.Tensor: Context vector of shape (..., seq_len_q, d_v).
    """
    d_k = query.size(-1)
    # (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Fill masked positions with a very small number (e.g., -1e9 or -inf)
        # so that their softmax probability becomes zero.
        scores = scores.masked_fill(mask, -1e9) # Use -1e9 as -inf can cause NaN with fp16

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # (..., seq_len_q, d_v)
    output = torch.matmul(attn_weights, value)
    return output


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.

    Attributes:
        d_model (int): The dimensionality of the model's embeddings.
        num_heads (int): The number of attention heads.
        d_k (int): Dimensionality of queries and keys per head.
        d_v (int): Dimensionality of values per head.
        W_q (nn.Linear): Linear layer for query projection.
        W_k (nn.Linear): Linear layer for key projection.
        W_v (nn.Linear): Linear layer for value projection.
        W_o (nn.Linear): Linear layer for output projection.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of K, Q per head
        self.d_v = d_model // num_heads # Dimension of V per head (typically d_k)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout_rate)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2) # Transpose for (batch_size, num_heads, seq_len, d_k)

    def _combine_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Combines multiple heads back into a single tensor.
        (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        """
        x = x.transpose(1, 2).contiguous() # (batch_size, seq_len, num_heads, d_k)
        return x.view(batch_size, -1, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, 1, 1, seq_len_k)
                                           or (batch_size, 1, seq_len_q, seq_len_k).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        batch_size = query.size(0)

        # 1) Linear projections and split into heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        query = self._split_heads(self.W_q(query), batch_size)
        key = self._split_heads(self.W_k(key), batch_size)
        value = self._split_heads(self.W_v(value), batch_size)

        # 2) Apply scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, num_heads, seq_len_q, d_v)
        attn_output = scaled_dot_product_attention(query, key, value, mask, self.dropout)

        # 3) Concatenate heads and apply final linear layer
        # (batch_size, num_heads, seq_len_q, d_v) -> (batch_size, seq_len_q, d_model)
        output = self.W_o(self._combine_heads(attn_output, batch_size))

        return output


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network (FFN).
    This consists of two linear transformations with a ReLU activation in between.

    Attributes:
        w_1 (nn.Linear): First linear layer.
        w_2 (nn.Linear): Second linear layer.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    Note: The paper's implementation applies layer normalization *after* the addition
    of the sublayer output and residual. This is known as "post-norm".

    Attributes:
        norm (nn.LayerNorm): Layer normalization layer.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, size: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Applies a sublayer, then dropout, residual connection, and layer normalization.

        Args:
            x (torch.Tensor): Input tensor (e.g., from previous sublayer).
            sublayer (Callable): A function representing the sublayer (e.g., MHA or FFN).

        Returns:
            torch.Tensor: Output tensor after sublayer, dropout, residual, and normalization.
        """
        # Post-norm: Add(x, Dropout(Sublayer(x))) then LayerNorm
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Encoder.
    Consists of a Multi-Head Self-Attention sublayer and a Position-wise Feed-Forward sublayer,
    each followed by a residual connection and layer normalization.

    Attributes:
        self_attn (MultiHeadAttention): Multi-head self-attention module.
        feed_forward (PositionwiseFeedForward): Position-wise feed-forward network.
        sublayer_1 (SublayerConnection): Sublayer connection for self-attention.
        sublayer_2 (SublayerConnection): Sublayer connection for feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer_1 = SublayerConnection(d_model, dropout_rate) # For self-attention
        self.sublayer_2 = SublayerConnection(d_model, dropout_rate) # For feed-forward

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous encoder layer or embeddings,
                              shape (batch_size, seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Source padding mask, shape (batch_size, 1, 1, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention sublayer
        x = self.sublayer_1(x, lambda x: self.self_attn(x, x, x, src_mask))
        # Position-wise feed-forward sublayer
        x = self.sublayer_2(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    The Transformer Encoder, consisting of an embedding layer, positional encoding,
    and a stack of `num_layers` EncoderLayers.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        layers (nn.ModuleList): Stack of EncoderLayer instances.
        norm (nn.LayerNorm): Final layer normalization (optional, often applied after last layer).
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_ff: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        # The original paper applies layer normalization after the final encoder layer
        # before passing to the decoder, but this is sometimes omitted or integrated.
        # We include it here for completeness as per some interpretations.
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the Encoder.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_len).
            src_mask (Optional[torch.Tensor]): Source padding mask, shape (batch_size, 1, 1, src_seq_len).

        Returns:
            torch.Tensor: Encoded memory tensor of shape (batch_size, src_seq_len, d_model).
        """
        # 1) Embeddings and positional encoding
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim) # Scale embeddings
        x = self.pos_encoder(x)

        # 2) Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x) # Apply final layer normalization


class DecoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Decoder.
    Consists of a Masked Multi-Head Self-Attention sublayer, a Multi-Head Encoder-Decoder Attention sublayer,
    and a Position-wise Feed-Forward sublayer, each followed by residual connections and layer normalization.

    Attributes:
        self_attn (MultiHeadAttention): Masked multi-head self-attention module.
        encoder_attn (MultiHeadAttention): Multi-head attention over encoder output.
        feed_forward (PositionwiseFeedForward): Position-wise feed-forward network.
        sublayer_1 (SublayerConnection): Sublayer connection for masked self-attention.
        sublayer_2 (SublayerConnection): Sublayer connection for encoder-decoder attention.
        sublayer_3 (SublayerConnection): Sublayer connection for feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.sublayer_1 = SublayerConnection(d_model, dropout_rate) # Masked self-attention
        self.sublayer_2 = SublayerConnection(d_model, dropout_rate) # Encoder-decoder attention
        self.sublayer_3 = SublayerConnection(d_model, dropout_rate) # Feed-forward

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the decoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer or embeddings,
                              shape (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output from the encoder (encoder_output),
                                   shape (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Source padding mask for encoder-decoder attention,
                                                shape (batch_size, 1, 1, src_seq_len).
            tgt_mask (Optional[torch.Tensor]): Target combined mask (padding + look-ahead) for
                                                masked self-attention, shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        # Masked self-attention sublayer (query, key, value are all from target)
        x = self.sublayer_1(x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # Encoder-decoder attention sublayer (query from target, key/value from encoder output)
        x = self.sublayer_2(x, lambda x: self.encoder_attn(x, memory, memory, src_mask))

        # Position-wise feed-forward sublayer
        x = self.sublayer_3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """
    The Transformer Decoder, consisting of an embedding layer, positional encoding,
    and a stack of `num_layers` DecoderLayers.

    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        layers (nn.ModuleList): Stack of DecoderLayer instances.
        norm (nn.LayerNorm): Final layer normalization.
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_ff: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer normalization

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the Decoder.

        Args:
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len).
            memory (torch.Tensor): Output from the encoder (encoder_output),
                                   shape (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Source padding mask for encoder-decoder attention,
                                                shape (batch_size, 1, 1, src_seq_len).
            tgt_mask (Optional[torch.Tensor]): Target combined mask (padding + look-ahead) for
                                                masked self-attention, shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        # 1) Embeddings and positional encoding
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim) # Scale embeddings
        x = self.pos_encoder(x)

        # 2) Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x) # Apply final layer normalization


class Generator(nn.Module):
    """
    The final linear layer and softmax for generating output probabilities.

    Attributes:
        proj (nn.Linear): Linear layer to project decoder output to vocabulary size.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the decoder output to the vocabulary space.

        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch_size, tgt_seq_len, d_model).

        Returns:
            torch.Tensor: Logits tensor of shape (