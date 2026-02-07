The Transformer model, as described in the "Attention Is All You Need" paper (Vaswani et al., 2017), is a groundbreaking architecture that relies entirely on attention mechanisms to process sequential data, eschewing traditional recurrent or convolutional layers. Below is a complete PyTorch implementation of this model, adhering to the specified requirements.

---

### PyTorch Implementation of the Transformer Model

This implementation provides a modular, well-documented, and type-hinted version of the Transformer architecture, including all its core components: Positional Encoding, Multi-Head Attention, Position-wise Feed-Forward Networks, residual connections with Layer Normalization, and the Encoder-Decoder structure.

```python
import torch
import torch.nn as nn
import math
from typing import Optional

# --- 1. Utility Functions for Masking ---

def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Creates a boolean mask to hide padding tokens in a sequence.

    Args:
        seq (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
        pad_idx (int): The index representing padding tokens.

    Returns:
        torch.Tensor: A boolean mask of shape (batch_size, 1, 1, seq_len)
                      where True indicates a padded element (to be masked).
    """
    # (batch_size, 1, 1, seq_len)
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(1)
    return mask

def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Creates a triangular mask to prevent attention to future tokens.
    Used in the decoder's self-attention mechanism.

    Args:
        seq_len (int): The length of the sequence.
        device (torch.device): The device (e.g., 'cpu', 'cuda') to create the mask on.

    Returns:
        torch.Tensor: A boolean mask of shape (1, 1, seq_len, seq_len)
                      where True indicates future elements (to be masked).
    """
    # Upper triangular matrix with ones, shape (seq_len, seq_len)
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # (1, 1, seq_len, seq_len)
    return look_ahead_mask.unsqueeze(0).unsqueeze(0)

# --- 2. Positional Encoding ---

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    The Transformer model itself has no inherent notion of order,
    so this encoding helps it understand the position of tokens in a sequence.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimension of the model (embedding dimension).
            dropout (float): The dropout rate to apply to the embeddings.
            max_len (int): The maximum expected sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension and register as a buffer
        # A buffer is a tensor that is not a model parameter but should be part of the model's state.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding and dropout to the input tensor.

        Args:
            x (torch.Tensor): Input tensor (embedded sequence) of shape
                              (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added, shape
                          (batch_size, seq_len, d_model).
        """
        # Add positional encoding to the input embeddings
        # The positional encoding is broadcasted across the batch dimension.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# --- 3. Scaled Dot-Product Attention ---

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (..., seq_len_q, d_k).
        key (torch.Tensor): Key tensor of shape (..., seq_len_k, d_k).
        value (torch.Tensor): Value tensor of shape (..., seq_len_v, d_v).
        mask (Optional[torch.Tensor]): Optional mask tensor of shape (..., seq_len_q, seq_len_k)
                                       or broadcastable, where True indicates positions to mask.
        dropout (Optional[nn.Dropout]): Optional dropout layer.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The attention output of shape (..., seq_len_q, d_v).
            - attention_weights (torch.Tensor): The attention weights of shape
                                                (..., seq_len_q, seq_len_k).
    """
    d_k = query.size(-1)
    # Scaled dot-product: (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Fill masked positions with a very large negative number
        # so they become 0 after softmax.
        scores = scores.masked_fill(mask, -1e9)

    attention_weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# --- 4. Multi-Head Attention ---

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention, which runs several attention mechanisms
    in parallel and concatenates their outputs. This allows the model to
    jointly attend to information from different representation subspaces
    at different positions.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            d_model (int): The dimension of the model (embedding dimension).
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of query/key per head
        self.d_v = d_model // num_heads  # Dimension of value per head

        # Linear layers for projecting Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final linear layer for output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
            mask (Optional[torch.Tensor]): Optional mask tensor for attention scores.
                                           Shape (batch_size, 1, seq_len_q, seq_len_k) or broadcastable.

        Returns:
            torch.Tensor: Output tensor after multi-head attention and projection,
                          shape (batch_size, seq_len_q, d_model).
        """
        batch_size = query.size(0)

        # 1) Project Q, K, V using linear layers
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2) Split into multiple heads and reshape
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k/d_v)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # 3) Apply scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, num_heads, seq_len_q, d_v)
        x, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)

        # 4) Concatenate heads and apply final linear layer
        # (batch_size, num_heads, seq_len_q, d_v) -> (batch_size, seq_len_q, num_heads * d_v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # (batch_size, seq_len_q, d_model) -> (batch_size, seq_len_q, d_model)
        output = self.w_o(x)
        return output

# --- 5. Position-wise Feed-Forward Network ---

class PositionWiseFeedForward(nn.Module):
    """
    A simple two-layer feed-forward network applied independently to each
    position (token) in the sequence. It's used after the attention mechanism.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the PositionWiseFeedForward layer.

        Args:
            d_model (int): The input and output dimension of the feed-forward network.
            d_ff (int): The dimension of the inner hidden layer.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feed-forward network to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        output = self.linear2(x)
        return output

# --- 6. Add & Normalize (Residual Connection and Layer Normalization) ---

class AddNorm(nn.Module):
    """
    Implements the "Add & Norm" step:
    1. A residual connection (adding the sub-layer input to its output).
    2. Followed by layer normalization.
    3. Dropout is applied to the sub-layer output before adding to the residual.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initializes the AddNorm layer.

        Args:
            d_model (int): The dimension of the model.
            dropout (float): The dropout rate to apply to the sublayer output.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Applies residual connection, dropout, and layer normalization.

        Args:
            x (torch.Tensor): The input to the sub-layer (for residual connection).
                              Shape (batch_size, seq_len, d_model).
            sublayer_output (torch.Tensor): The output of the sub-layer (e.g., attention or FFN).
                                            Shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Normalized output after residual connection and dropout,
                          shape (batch_size, seq_len, d_model).
        """
        # Add residual connection and apply dropout to sublayer output
        return self.norm(x + self.dropout(sublayer_output))

# --- 7. Encoder Layer ---

class EncoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Encoder.
    Consists of a Multi-Head Self-Attention sub-layer and a Position-wise Feed-Forward sub-layer,
    each followed by a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes an EncoderLayer.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the EncoderLayer.

        Args:
            src (torch.Tensor): Input tensor from the previous layer or embedded source sequence.
                                Shape (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Mask for the source sequence (e.g., padding mask).
                                                Shape (batch_size, 1, 1, src_seq_len) or broadcastable.

        Returns:
            torch.Tensor: Output tensor of the EncoderLayer,
                          shape (batch_size, src_seq_len, d_model).
        """
        # Self-attention sub-layer
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = self.add_norm1(src, attn_output)

        # Feed-forward sub-layer
        ff_output = self.feed_forward(src)
        src = self.add_norm2(src, ff_output)
        
        return src

# --- 8. Decoder Layer ---

class DecoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Decoder.
    Consists of three sub-layers:
    1. Masked Multi-Head Self-Attention (to prevent attending to future tokens).
    2. Multi-Head Encoder-Decoder Attention (attends to encoder output).
    3. Position-wise Feed-Forward.
    Each sub-layer is followed by a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a DecoderLayer.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs a forward pass through the DecoderLayer.

        Args:
            tgt (torch.Tensor): Input tensor from the previous decoder layer or embedded target sequence.
                                Shape (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output tensor from the Encoder stack.
                                   Shape (batch_size, src_seq_len, d_model).
            tgt_mask (Optional[torch.Tensor]): Mask for the target self-attention. Combines padding
                                                and look-ahead masks.
                                                Shape (batch_size, 1, tgt_seq_len, tgt_seq_len) or broadcastable.
            memory_mask (Optional[torch.Tensor]): Mask for the encoder output (memory), typically a padding mask.
                                                  Shape (batch_size, 1, 1, src_seq_len) or broadcastable.

        Returns:
            torch.Tensor: Output tensor of the DecoderLayer,
                          shape (batch_size, tgt_seq_len, d_model).
        """
        # Masked Multi-Head Self-Attention sub-layer
        attn1_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.add_norm1(tgt, attn1_output)

        # Multi-Head Encoder-Decoder Attention sub-layer
        # Query comes from the decoder, Key and Value come from the encoder output (memory)
        attn2_output = self.encoder_decoder_attn(tgt, memory, memory, mask=memory_mask)
        tgt = self.add_norm2(tgt, attn2_output)

        # Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(tgt)
        tgt = self.add_norm3(tgt, ff_output)

        return tgt

# --- 9. Encoder Stack ---

class Encoder(nn.Module):
    """
    The Encoder stack of the Transformer.
    Consists of an input embedding layer, positional encoding, and a stack of N EncoderLayers.
    """
    def __init__(
        self,
        src_vocab_size: int,
        d_model: int,
        num_encoder_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0
    ):
        """
        Initializes the Encoder stack.

        Args:
            src_vocab_size (int): The size of the source vocabulary.
            d_model (int): The dimension of the model (embedding dimension).
            num_encoder_layers (int): The number of identical EncoderLayers to stack.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner feed-forward layer.
            dropout (float): The dropout rate.
            max_len (int): The maximum expected sequence length for positional encoding.
            pad_idx (int): The index for padding tokens in the source vocabulary.
        """
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model) # Final layer norm after the stack

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs a forward pass through the Encoder stack.

        Args:
            src (torch.Tensor): Source input sequence tensor of shape (batch_