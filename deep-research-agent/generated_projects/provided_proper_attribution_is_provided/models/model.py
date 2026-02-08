The Transformer model, as introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequence transduction by relying entirely on attention mechanisms, doing away with recurrent or convolutional layers. Below is a complete PyTorch implementation adhering to the specified requirements.

This implementation covers:
1.  **Scaled Dot-Product Attention**: The fundamental building block.
2.  **Multi-Head Attention**: Extends dot-product attention to multiple representation subspaces.
3.  **Position-wise Feed-Forward Networks**: Simple two-layer FFN applied independently to each position.
4.  **Positional Encoding**: Injects information about the relative or absolute position of tokens in the sequence.
5.  **Residual Connections and Layer Normalization**: Crucial for training deep networks.
6.  **Encoder Layer**: Comprises multi-head self-attention and a feed-forward network.
7.  **Decoder Layer**: Comprises masked multi-head self-attention, encoder-decoder multi-head attention, and a feed-forward network.
8.  **Encoder**: Stacks multiple Encoder Layers.
9.  **Decoder**: Stacks multiple Decoder Layers.
10. **Transformer**: The complete end-to-end model, including input/output embeddings and final linear projection.
11. **Masking Utilities**: For padding and look-ahead.

---

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in the paper.
    This module adds positional information to the input embeddings, allowing
    the model to account for the order of tokens in a sequence.

    The positional encoding uses sine and cosine functions of different frequencies.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attributes:
        d_model (int): The dimensionality of the input embeddings.
        max_len (int): The maximum sequence length for which to generate encodings.
        dropout (nn.Dropout): Dropout layer to apply to the combined embeddings and PE.
        pe (torch.Tensor): Pre-computed positional encoding matrix.
    """
    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add a batch dimension
        self.register_buffer('pe', pe) # Register as a buffer so it's not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor with positional encoding added,
                          shape (batch_size, seq_len, d_model).
        """
        # x has shape (batch_size, seq_len, d_model)
        # self.pe has shape (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Computes the Scaled Dot-Product Attention as described in the paper.
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Attributes:
        dropout (nn.Dropout): Dropout layer to apply to attention weights.
    """
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, d_v).
                                  Note: seq_len_k must be equal to seq_len_v.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, 1, seq_len_k)
                                           or (batch_size, 1, seq_len_q, seq_len_k).
                                           Used to mask out future tokens (decoder self-attention)
                                           or padding tokens. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output of the attention mechanism, shape (batch_size, num_heads, seq_len_q, d_v).
                - Attention weights, shape (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        d_k = query.size(-1)
        # (batch_size, num_heads, seq_len_q, d_k) @ (batch_size, num_heads, d_k, seq_len_k)
        # -> (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask has shape (batch_size, 1, 1, seq_len_k) for padding or
            # (batch_size, 1, seq_len_q, seq_len_k) for look-ahead
            # Where mask is 0 (padding/future token), set scores to a very small number
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, seq_len_q, seq_len_k) @ (batch_size, num_heads, seq_len_k, d_v)
        # -> (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention.
    It runs multiple scaled dot-product attention layers in parallel and
    then concatenates their outputs, which are then linearly transformed.

    Attributes:
        d_model (int): Dimensionality of the model's embeddings.
        num_heads (int): Number of attention heads.
        d_k (int): Dimensionality of keys and queries per head (d_model // num_heads).
        d_v (int): Dimensionality of values per head (d_model // num_heads).
        query_projection (nn.Linear): Linear projection for queries.
        key_projection (nn.Linear): Linear projection for keys.
        value_projection (nn.Linear): Linear projection for values.
        output_projection (nn.Linear): Final linear projection after concatenating heads.
        attention (ScaledDotProductAttention): Instance of scaled dot-product attention.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of keys/queries per head
        self.d_v = d_model // num_heads # Dimension of values per head

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout_rate)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits the last dimension of the tensor into (num_heads, d_k or d_v).
        Rearranges dimensions to (batch_size, num_heads, seq_len, d_k or d_v).
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines the outputs of multiple attention heads back into a single tensor.
        Rearranges dimensions from (batch_size, num_heads, seq_len, d_v)
        to (batch_size, seq_len, d_model).
        """
        batch_size, num_heads, seq_len, d_v = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
                                  Note: seq_len_k must be equal to seq_len_v.
            mask (torch.Tensor, optional): Mask tensor. Same shape requirements as
                                           ScaledDotProductAttention.forward(). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Output of the multi-head attention, shape (batch_size, seq_len_q, d_model).
                - Attention weights (average across heads if needed, or raw).
                  This implementation returns raw weights from the last head for simplicity.
                  Shape (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        batch_size = query.size(0)

        # 1. Project Q, K, V linearly
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query_proj = self.query_projection(query)
        key_proj = self.key_projection(key)
        value_proj = self.value_projection(value)

        # 2. Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k/v)
        query_heads = self._split_heads(query_proj)
        key_heads = self._split_heads(key_proj)
        value_heads = self._split_heads(value_proj)

        # 3. Apply scaled dot-product attention
        # Mask needs to be broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            # Add num_heads dimension to mask if it's not already there
            if mask.dim() == 3: # (batch_size, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1) # (batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 2: # (batch_size, seq_len_k) for padding
                mask = mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len_k)

        attn_output, attn_weights = self.attention(query_heads, key_heads, value_heads, mask)

        # 4. Concatenate heads and apply final linear projection
        # (batch_size, num_heads, seq_len_q, d_v) -> (batch_size, seq_len_q, d_model)
        output = self._combine_heads(attn_output)
        output = self.output_projection(output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN).
    This network consists of two linear transformations with a ReLU activation in between.
    FFN(x) = max(0, x W1 + b1) W2 + b2

    Attributes:
        w_1 (nn.Linear): First linear layer.
        w_2 (nn.Linear): Second linear layer.
        dropout (nn.Dropout): Dropout layer applied after the second linear layer.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feed-forward network to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.dropout(self.w_2(F.relu(self.w_1(x))))


class AddNorm(nn.Module):
    """
    Implements the "Add & Norm" component:
    It applies a residual connection followed by Layer Normalization.
    x + Sublayer(x) then LayerNorm(x + Sublayer(x))

    Attributes:
        norm (nn.LayerNorm): Layer normalization module.
        dropout (nn.Dropout): Dropout layer applied to the sublayer output.
    """
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Applies residual connection and layer normalization.

        Args:
            x (torch.Tensor): The input to the sublayer, to be used for the residual connection.
                              Shape (batch_size, seq_len, d_model).
            sublayer_output (torch.Tensor): The output of the sublayer (e.g., MHA or FFN).
                                            Shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The normalized output after residual connection.
                          Shape (batch_size, seq_len, d_model).
        """
        return self.norm(x + self.dropout(sublayer_output))


class EncoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Encoder.
    It consists of two sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    Each sub-layer is followed by a residual connection and layer normalization.

    Attributes:
        self_attn (MultiHeadAttention): Multi-head self-attention module.
        feed_forward (PositionwiseFeedForward): Position-wise feed-forward network.
        add_norm1 (AddNorm): Add&Norm for the self-attention sub-layer.
        add_norm2 (AddNorm): Add&Norm for the feed-forward sub-layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.add_norm1 = AddNorm(d_model, dropout_rate)
        self.add_norm2 = AddNorm(d_model, dropout_rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through one encoder layer.

        Args:
            x (torch.Tensor): Input tensor to the encoder layer, shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, optional): Mask for the source sequence (e.g., padding mask).
                                               Shape (batch_size, 1, 1, seq_len) or
                                               (batch_size, 1, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of the encoder layer, shape (batch_size, seq_len, d_model).
        """
        # Multi-Head Self-Attention sub-layer
        # Query, Key, Value are all the same (self-attention)
        attn_output, _ = self.self_attn(x, x, x, mask=src_mask)
        x = self.add_norm1(x, attn_output)

        # Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)
        return x


class DecoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Decoder.
    It consists of three sub-layers:
    1. Masked Multi-Head Self-Attention (for target sequence)
    2. Multi-Head Encoder-Decoder Attention (attends to encoder output)
    3. Position-wise Feed-Forward Network

    Each sub-layer is followed by a residual connection and layer normalization.

    Attributes:
        self_attn (MultiHeadAttention): Masked multi-head self-attention module for target sequence.
        encoder_decoder_attn (MultiHeadAttention): Multi-head attention module attending to encoder output.
        feed_forward (PositionwiseFeedForward): Position-wise feed-forward network.
        add_norm1 (AddNorm): Add&Norm for masked self-attention.
        add_norm2 (AddNorm): Add&Norm for encoder-decoder attention.
        add_norm3 (AddNorm): Add&Norm for feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.add_norm1 = AddNorm(d_model, dropout_rate)
        self.add_norm2 = AddNorm(d_model, dropout_rate)
        self.add_norm3 = AddNorm(d_model, dropout_rate)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through one decoder layer.

        Args:
            x (torch.Tensor): Input tensor to the decoder layer (target sequence embeddings),
                              shape (batch_size, tgt_seq_len, d_model).
            enc_output (torch.Tensor): Output from the encoder stack,
                                       shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor, optional): Mask for the source sequence (encoder output),
                                               used in encoder-decoder attention.
                                               Shape (batch_size, 1, 1, src_seq_len). Defaults to None.
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (decoder self-attention),
                                               combines padding and look-ahead masks.
                                               Shape (batch_size, 1, tgt_seq_len, tgt_seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of the decoder layer, shape (batch_size, tgt_seq_len, d_model).
        """
        # Masked Multi-Head Self-Attention sub-layer (on target sequence)
        # Query, Key, Value are all the same (x)
        self_attn_output, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.add_norm1(x, self_attn_output)

        # Multi-Head Encoder-Decoder Attention sub-layer
        # Query is from decoder (x), Key/Value are from encoder output
        enc_dec_attn_output, _ = self.encoder_decoder_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.add_norm2(x, enc_dec_attn_output)

        # Position-wise Feed-Forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)
        return x


class Encoder(nn.Module):
    """
    The Transformer Encoder stack.
    It comprises an input embedding layer, positional encoding, and a stack of N identical EncoderLayers.

    Attributes:
        embedding (nn.Embedding): Token embedding layer for the source vocabulary.
        positional_encoding (PositionalEncoding): Positional encoding module.
        layers (nn.ModuleList): A list of N EncoderLayer instances.
        norm (nn.LayerNorm): Final layer normalization (optional, often used in BERT-like models,
                             but not explicitly in original Transformer paper after encoder stack).
                             Included here for potential robustness.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Optional: Layer norm after the final encoder layer

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through the encoder stack.

        Args:
            src (torch.Tensor): Source input sequence of token IDs, shape (batch_size, src_seq_len).
            src_mask (torch.Tensor, optional): Padding mask for the source sequence.
                                               Shape (batch_size, 1, 1, src_seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor from the encoder stack,
                          shape (batch_size, src_seq_len, d_model).
        """
        # 1. Input Embedding + Positional Encoding
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim) # Scale embeddings
        x = self.positional_encoding(x)

        # 2. Pass through N encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x) # Apply final layer norm


class Decoder(nn.Module):
    """
    The Transformer Decoder stack.
    It comprises an output embedding layer, positional encoding, and a stack of N identical DecoderLayers.

    Attributes:
        embedding (nn.Embedding): Token embedding layer for the target vocabulary.
        positional_encoding (PositionalEncoding): Positional encoding module.
        layers (nn.ModuleList): A list of N DecoderLayer instances.
        norm (nn.LayerNorm): Final layer normalization.
    """
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int,
                 d_ff: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer norm

    def forward(self, tgt: torch.Tensor, enc_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through the decoder stack.

        Args:
            tgt (torch.Tensor): Target input sequence of token IDs (shifted right),
                                shape (batch_size, tgt_seq_len).
            enc_output (torch.Tensor): Output from the encoder stack,
                                       shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor, optional): Padding mask for the source sequence,
                                               used in encoder-decoder attention.
                                               Shape (batch_size, 1, 1, src_seq_len). Defaults to None.
            tgt_mask (torch.Tensor, optional): Mask for the target sequence (combines padding and look-ahead).
                                               Shape (batch_size, 1, tgt_seq_len, tgt_seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor from the decoder stack,
                          shape (batch_size, tgt_seq_len, d_model).
        """
        # 1. Input Embedding + Positional Encoding
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim) # Scale embeddings
        x = self.positional_encoding(x)

        # 2. Pass through N decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.norm(x) # Apply final layer norm


class Transformer(nn.Module):
    """
    The complete Transformer model for sequence-to-sequence tasks.
    It consists of an Encoder, a Decoder, and a final linear layer for output prediction.

    Attributes:
        encoder (Encoder): The encoder stack.
        decoder (Decoder): The decoder stack.
        output_projection (nn.Linear): Final linear layer to project decoder output
                                       to vocabulary size for token prediction.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048,
                 dropout_rate: float = 0.1, max_len: int = 5000,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads,
                               d_ff, dropout_rate, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads,
                               d_ff, dropout_rate, max_len)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights() # Initialize model parameters

    def _init_weights(self):
        """
        Initializes model parameters with Xavier uniform distribution.
        This helps with training stability.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Creates a padding mask for a given sequence.
        Masks out padding tokens (specified by self.pad_idx).

        Args:
            seq (torch.Tensor): Input sequence of token IDs, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Padding mask, shape (batch_size, 1, 1, seq_len),
                          where 1 indicates a non-padding token and 0 indicates padding.
        """
        # (batch_size, 1, 1, seq_len)
        mask = (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.int() # Convert to int for compatibility with masked_fill


    def _create_look_ahead_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a look-ahead mask to prevent attention to future tokens in the target sequence.

        Args:
            seq_len (int): Length of the sequence for which to create the mask.
            device (torch.device): The device (