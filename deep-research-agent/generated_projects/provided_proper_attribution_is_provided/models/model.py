Okay, based on the description of the Transformer model from "Attention Is All You Need," here is a complete PyTorch implementation following best practices, including detailed docstrings, type hints, and modularity.

The Transformer architecture consists of:
1.  **Input Embeddings:** Token embeddings combined with positional encodings.
2.  **Encoder:** A stack of identical layers, each with a Multi-Head Self-Attention mechanism and a Position-wise Feed-Forward Network, both followed by residual connections and Layer Normalization.
3.  **Decoder:** A stack of identical layers, each with a Masked Multi-Head Self-Attention mechanism, an Encoder-Decoder Multi-Head Attention mechanism, and a Position-wise Feed-Forward Network, all followed by residual connections and Layer Normalization.
4.  **Output Layer:** A final linear layer and softmax for sequence prediction.

---

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# --- Utility Functions for Masking ---

def generate_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Generates a padding mask for a batch of sequences.

    Args:
        seq (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
        pad_idx (int): The index of the padding token.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, 1, 1, seq_len)
                      where True indicates a padding token (should be ignored).
                      This mask is broadcastable to attention scores.
    """
    # Returns True for padding tokens, False otherwise
    padding_mask = (seq == pad_idx).unsqueeze(1).unsqueeze(1)
    return padding_mask # (batch_size, 1, 1, seq_len)

def generate_subsequent_mask(seq_len: int) -> torch.Tensor:
    """
    Generates a causal (subsequent) mask for self-attention.
    Ensures that a token at position 'i' can only attend to tokens
    at positions 'j <= i'.

    Args:
        seq_len (int): The length of the sequence.

    Returns:
        torch.Tensor: A boolean tensor of shape (1, 1, seq_len, seq_len)
                      where True indicates positions that should be masked (future tokens).
    """
    # tril creates a lower triangular matrix
    # (seq_len, seq_len)
    subsequent_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return subsequent_mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)


# --- Core Transformer Components ---

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in the paper.
    This adds information about the absolute position of tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            dropout (float): Dropout rate to apply to the output.
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
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe) # 'pe' is not a learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encodings added,
                          shape (batch_size, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1)].detach() # Add positional encoding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    Each head performs scaled dot-product attention independently,
    and their outputs are concatenated and linearly transformed.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            n_heads (int): The number of attention heads.
            dropout (float): Dropout rate to apply to the output.

        Raises:
            ValueError: If d_model is not divisible by n_heads.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # Dimension of K, Q, V for each head

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, n_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines the outputs from multiple heads back into a single tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_heads, seq_len, d_k).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

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
            query (torch.Tensor): Query tensor of shape (batch_size, query_seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, key_seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, value_seq_len, d_model).
                                  (key_seq_len and value_seq_len must be the same).
            mask (Optional[torch.Tensor]): Attention mask of shape
                                           (batch_size, 1, query_seq_len, key_seq_len) or
                                           (1, 1, query_seq_len, key_seq_len).
                                           True indicates positions to be masked (set to -inf).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_seq_len, d_model).
        """
        # 1. Project Q, K, V
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        queries = self.query_proj(query)
        keys = self.key_proj(key)
        values = self.value_proj(value)

        # 2. Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # 3. Scaled Dot-Product Attention
        # (batch_size, n_heads, query_seq_len, d_k) @ (batch_size, n_heads, d_k, key_seq_len)
        # -> (batch_size, n_heads, query_seq_len, key_seq_len)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Apply mask: fill masked positions with a very small number (-inf effectively)
            # Mask is typically (batch_size, 1, 1, key_seq_len) or (1, 1, query_seq_len, key_seq_len)
            # The broadcasting will handle this.
            scores = scores.masked_fill(mask, -1e9) # Use -1e9 instead of float('-inf') for stability

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights) # Dropout on attention weights

        # (batch_size, n_heads, query_seq_len, key_seq_len) @ (batch_size, n_heads, value_seq_len, d_k)
        # -> (batch_size, n_heads, query_seq_len, d_k)
        context = torch.matmul(attention_weights, values)

        # 4. Combine heads and project back
        # (batch_size, n_heads, query_seq_len, d_k) -> (batch_size, query_seq_len, d_model)
        output = self._combine_heads(context)
        output = self.out_proj(output)

        return output


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network (FFN).
    Consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the PositionwiseFeedForward layer.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            d_ff (int): The dimensionality of the inner-layer (feed-forward dimension).
            dropout (float): Dropout rate to apply to the output.
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.dropout(self.w_2(F.relu(self.w_1(x))))


class AddNorm(nn.Module):
    """
    Implements the Add & Norm layer, which combines a residual connection
    with Layer Normalization.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Initializes the AddNorm layer.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            dropout (float): Dropout rate to apply to the output of the sub-layer.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Applies residual connection and layer normalization.

        Args:
            x (torch.Tensor): The input to the sub-layer (for residual connection),
                              shape (batch_size, seq_len, d_model).
            sublayer_output (torch.Tensor): The output from the sub-layer (e.g., MHA or FFN),
                                            shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Normalized output with residual connection,
                          shape (batch_size, seq_len, d_model).
        """
        # Apply dropout to the sublayer output BEFORE adding to the residual
        return self.norm(x + self.dropout(sublayer_output))


# --- Encoder and Decoder Layers ---

class EncoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer Encoder.
    Consists of a Multi-Head Self-Attention sub-layer and a Position-wise Feed-Forward
    sub-layer, each followed by residual connections and Layer Normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes an EncoderLayer.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner-layer of the FFN.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass for an Encoder Layer.

        Args:
            src (torch.Tensor): Input tensor from the previous layer (or embeddings),
                                shape (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Source padding mask of shape
                                                (batch_size, 1, 1, src_seq_len).
                                                True indicates positions to be masked.

        Returns:
            torch.Tensor: Output tensor of the encoder layer,
                          shape (batch_size, src_seq_len, d_model).
        """
        # Self-attention sub-layer
        # Query, Key, Value are all 'src'
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = self.add_norm_1(src, attn_output) # Add & Norm

        # Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(src)
        src = self.add_norm_2(src, ff_output) # Add & Norm

        return src


class DecoderLayer(nn.Module):
    """
    Implements a single layer of the Transformer Decoder.
    Consists of a Masked Multi-Head Self-Attention sub-layer,
    a Multi-Head Encoder-Decoder Attention sub-layer, and a Position-wise Feed-Forward
    sub-layer, each followed by residual connections and Layer Normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a DecoderLayer.

        Args:
            d_model (int): The dimensionality of the model's embeddings.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the inner-layer of the FFN.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)
        self.add_norm_3 = AddNorm(d_model, dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass for a Decoder Layer.

        Args:
            tgt (torch.Tensor): Input tensor from the previous decoder layer (or embeddings),
                                shape (batch_size, tgt_seq_len, d_model).
            encoder_output (torch.Tensor): Output from the encoder stack,
                                           shape (batch_size, src_seq_len, d_model).
            tgt_mask (Optional[torch.Tensor]): Target (decoder) padding and causal mask,
                                                shape (batch_size, 1, tgt_seq_len, tgt_seq_len) or
                                                (1, 1, tgt_seq_len, tgt_seq_len).
                                                True indicates positions to be masked.
            src_mask (Optional[torch.Tensor]): Source (encoder) padding mask,
                                                shape (batch_size, 1, 1, src_seq_len).
                                                True indicates positions to be masked.

        Returns:
            torch.Tensor: Output tensor of the decoder layer,
                          shape (batch_size, tgt_seq_len, d_model).
        """
        # Masked self-attention sub-layer (attends to previous decoder outputs)
        # Query, Key, Value are all 'tgt'
        self_attn_output = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.add_norm_1(tgt, self_attn_output)

        # Encoder-Decoder attention sub-layer (attends to encoder outputs)
        # Query is 'tgt', Key and Value are 'encoder_output'
        enc_dec_attn_output = self.encoder_attn(tgt, encoder_output, encoder_output, mask=src_mask)
        tgt = self.add_norm_2(tgt, enc_dec_attn_output)

        # Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(tgt)
        tgt = self.add_norm_3(tgt, ff_output)

        return tgt


# --- Encoder and Decoder Stacks ---

class Encoder(nn.Module):
    """
    The Encoder stack of the Transformer.
    Comprises an input embedding layer, positional encoding, and a stack of N identical EncoderLayers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        pad_idx: int
    ):
        """
        Initializes the Encoder.

        Args:
            vocab_size (int): Size of the source vocabulary.
            d_model (int): Dimensionality of model embeddings.
            n_layers (int): Number of identical EncoderLayers.
            n_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the FFN's inner layer.
            dropout (float): Dropout rate.
            pad_idx (int): Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer norm as per some implementations

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass for the Encoder stack.

        Args:
            src (torch.Tensor): Input source sequence tensor, shape (batch_size, src_seq_len).
            src_mask (Optional[torch.Tensor]): Source padding mask,
                                                shape (batch_size, 1, 1, src_seq_len).
                                                True indicates positions to be masked.

        Returns:
            torch.Tensor: Output tensor from the encoder stack,
                          shape (batch_size, src_seq_len, d_model).
        """
        # Apply token embedding and scale by sqrt(d_model)
        # The paper scales embeddings by sqrt(d_model) before adding PE
        src = self.token_embedding(src) * math.sqrt(self.token_embedding.embedding_dim)
        src = self.positional_encoding(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return self.norm(src) # Apply final layer norm


class Decoder(nn.Module):
    """
    The Decoder stack of the Transformer.
    Comprises an input embedding layer, positional encoding, and a stack of N identical DecoderLayers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        pad_idx: int
    ):
        """
        Initializes the Decoder.

        Args:
            vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimensionality of model embeddings.
            n_layers (int): Number of identical DecoderLayers.
            n_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the FFN's inner layer.
            dropout (float): Dropout rate.
            pad_idx (int): Index of the padding token in the vocabulary.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer norm

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs the forward pass for the Decoder stack.

        Args:
            tgt (torch.Tensor): Input target sequence tensor, shape (batch_size, tgt_seq_len).
            encoder_output (torch.Tensor): Output from the encoder stack,
                                           shape (batch_size, src_seq_len, d_model).
            tgt_mask (Optional[torch.Tensor]): Target (decoder) padding and causal mask,
                                                shape (batch_size, 1, tgt_seq_len, tgt_seq_len) or
                                                (1, 1, tgt_seq_len, tgt_seq_len).
                                                True indicates positions to be masked.
            src_mask (Optional[torch.Tensor]): Source (encoder) padding mask,
                                                shape (batch_size, 1, 1, src_seq_len).
                                                True indicates positions to be masked.

        Returns:
            torch.Tensor: Output tensor from the decoder stack,
                          shape (batch_size, tgt_seq_len, d_model).
        """
        # Apply token embedding and scale by sqrt(d_model)
        tgt = self.token_embedding(tgt) * math.sqrt(self.token_embedding.embedding_dim)
        tgt = self.positional_encoding(tgt)

        for layer in self.layers:
            tgt = layer(tgt, encoder_output, tgt_mask, src_mask)

        return self.norm(tgt) # Apply final layer norm


# --- Full Transformer Model ---

class Transformer(nn.Module):
    """
    The full Transformer model for sequence-to-sequence tasks.
    Combines an Encoder and a Decoder, with a final linear layer for output prediction.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0
    ):
        """
        Initializes the Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimensionality of model embeddings (default: 512).
            n_layers (int): Number of identical Encoder/DecoderLayers (default: 6).
            n_heads (int): Number of attention heads (default: 8).
            d_ff (int): Dimensionality of the FFN's inner layer (default: 2048).
            dropout (float): Dropout rate (default: 0.1).
            src_pad_idx (int): Index of the padding token in the source vocabulary.
            tgt_pad_idx (int): Index of the padding token in the target vocabulary.
        """
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        self.encoder = Encoder(
            src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, src_pad_idx
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, tgt_pad_idx
        )
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters with Xavier uniform as per the paper
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates the source padding mask.

        Args:
            src (torch.Tensor): Source sequence tensor, shape (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Source padding mask, shape (batch_size, 1, 1, src_seq_len).
        """
        return generate_padding_mask(src, self.src_pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Creates the target padding and subsequent (causal) mask.

        Args:
            tgt (torch.Tensor): Target sequence tensor, shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Combined target mask, shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
        """
        tgt_pad_mask = generate_padding_mask(tgt, self.tgt_pad_idx)
        tgt_sub_mask = generate_subsequent_mask(tgt.size(1)).to(tgt.device)
        # Combine masks: True if either padding or subsequent mask is True
        # The padding mask is (batch_size, 1, 1, tgt_seq_len)
        # The subsequent mask is (1, 1, tgt_seq_len, tgt_seq_len)
        # When combined, they broadcast correctly to (batch_size, 1, tgt_seq_len, tgt_seq_len)
        return tgt_pad_mask | tgt_sub_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the forward pass for the Transformer model.

        Args:
            src (torch.Tensor): Source sequence tensor, shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target sequence tensor, shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Logits for the target vocabulary,
                          shape (batch_size, tgt_seq_