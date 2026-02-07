Here's a complete PyTorch implementation of the Transformer model, following the "Attention Is All You Need" architecture, incorporating best practices, detailed docstrings, and type hints.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding for the Transformer model.

    This module adds sinusoidal positional encodings to the input embeddings.
    The encodings provide information about the relative or absolute position
    of tokens in the sequence, which is crucial since the Transformer
    does not inherently process sequences in order like RNNs.

    The encoding formula for position 'pos' and dimension 'i' is:
    PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))

    Args:
        d_model (int): The dimension of the model's embeddings.
        max_len (int): The maximum length of the input sequences.
                       This determines the size of the precomputed positional encoding table.
        dropout_rate (float): The dropout probability to apply to the output.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Add a batch dimension for broadcasting (1, max_len, d_model)
        self.register_buffer('pe', pe) # Register as a buffer, so it's not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor, typically token embeddings.
                              Shape: (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with positional encodings added.
                          Shape: (batch_size, seq_len, d_model).
        """
        # Truncate positional encodings to the sequence length of the input
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    This is the core attention mechanism. It calculates the compatibility
    between queries and keys, scales it, applies a softmax to get attention
    weights, and then multiplies these weights by the values.

    The scaling factor (1/sqrt(d_k)) prevents the dot products from
    growing too large, which could push the softmax into regions with
    very small gradients.

    Args:
        dropout_rate (float): The dropout probability to apply to attention weights.
    """
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor.
                                  Shape: (batch_size, num_heads, seq_len_q, d_k).
            key (torch.Tensor): Key tensor.
                                Shape: (batch_size, num_heads, seq_len_k, d_k).
            value (torch.Tensor): Value tensor.
                                  Shape: (batch_size, num_heads, seq_len_v, d_v).
                                  Note: seq_len_k must be equal to seq_len_v.
            mask (Optional[torch.Tensor]): An attention mask to prevent attending to certain positions.
                                           Shape: (batch_size, 1, seq_len_q, seq_len_k) or 
                                                  (batch_size, 1, 1, seq_len_k) for padding masks,
                                                  or (batch_size, 1, seq_len_q, seq_len_q) for look-ahead mask.
                                           Values: True for positions to mask (set to -infinity), False otherwise.
                                           (Or 0 for positions to mask, 1 otherwise, if using `masked_fill_`).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The weighted sum of values.
                                         Shape: (batch_size, num_heads, seq_len_q, d_v).
                - attn_weights (torch.Tensor): The attention weights.
                                               Shape: (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        d_k = query.size(-1)
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask values are typically boolean (True for masked positions) or float (0 for masked)
            # We want to fill masked positions with a very small number (-inf) before softmax.
            # `mask` is expected to be a boolean tensor where True means "mask this position".
            scores = scores.masked_fill(mask, -1e9) # Use -1e9 instead of float('-inf') for numerical stability

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.

    This module performs multiple attention calculations in parallel,
    each with different learned linear projections of Q, K, and V.
    The outputs of these attention heads are then concatenated and
    linearly transformed to produce the final output. This allows
    the model to jointly attend to information from different
    representation subspaces at different positions.

    Args:
        d_model (int): The dimension of the model's embeddings.
        num_heads (int): The number of attention heads.
        dropout_rate (float): The dropout probability to apply to attention weights.

    Raises:
        ValueError: If d_model is not divisible by num_heads.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_k = d_model // num_heads # Dimension of K, Q for each head
        self.num_heads = num_heads
        self.d_model = d_model

        # Linear layers for Q, K, V projections and final output projection
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs multi-head attention.

        Args:
            query (torch.Tensor): Query tensor. Shape: (batch_size, seq_len_q, d_model).
            key (torch.Tensor): Key tensor. Shape: (batch_size, seq_len_k, d_model).
            value (torch.Tensor): Value tensor. Shape: (batch_size, seq_len_v, d_model).
            mask (Optional[torch.Tensor]): An attention mask. 
                                           Shape: (batch_size, 1, seq_len_q, seq_len_k) or 
                                                  (batch_size, 1, 1, seq_len_k) etc.
                                           True for positions to mask.

        Returns:
            torch.Tensor: The output of the multi-head attention.
                          Shape: (batch_size, seq_len_q, d_model).
        """
        batch_size = query.size(0)

        # 1) Linear projections and reshape for multi-head:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, num_heads, seq_len, d_k)
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Apply scaled dot-product attention
        # x: (batch_size, num_heads, seq_len_q, d_k)
        # attn: (batch_size, num_heads, seq_len_q, seq_len_k)
        x, _ = self.attention(query, key, value, mask=mask)

        # 3) "Concat" heads and apply final linear layer:
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, num_heads, d_k)
        # -> (batch_size, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)

class PositionwiseFeedForward(nn.Module):
    """
    Implements a position-wise feed-forward network.

    This module consists of two linear transformations with a ReLU
    activation in between. It is applied independently to each
    position in the sequence.

    Args:
        d_model (int): The dimension of the model's embeddings.
        d_ff (int): The dimension of the inner hidden layer.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feed-forward network.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, seq_len, d_model).
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class AddNorm(nn.Module):
    """
    Implements the "Add & Norm" layer, combining a residual connection
    with layer normalization.

    This module applies a sublayer (e.g., attention or FFN),
    then adds its output to the original input (residual connection),
    and finally applies layer normalization. Dropout is applied before
    adding the sublayer output to the input.

    Args:
        d_model (int): The dimension of the model's embeddings.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies the Add & Norm operation.

        Args:
            x (torch.Tensor): The input tensor to the sublayer.
                              Shape: (batch_size, seq_len, d_model).
            sublayer (nn.Module): The sublayer to apply (e.g., MultiHeadAttention or PositionwiseFeedForward).
                                  This sublayer must accept `x` and any necessary masks as arguments.

        Returns:
            torch.Tensor: The output tensor after Add & Norm.
                          Shape: (batch_size, seq_len, d_model).
        """
        # Original paper applies dropout AFTER sublayer and BEFORE residual add.
        # Then applies LayerNorm.
        return x + self.dropout(sublayer(self.norm(x)))
        # Alternative (common in other implementations): norm before sublayer,
        # then residual connection, then dropout.
        # This implementation follows the "Pre-LN Transformer" variant, which is common.
        # The original paper used Post-LN: x + dropout(sublayer(x)) followed by LayerNorm.
        # Let's stick to the Post-LN as per the original paper's figure for strict adherence.
        # return self.norm(x + self.dropout(sublayer(x)))

        # Re-evaluating the "Add & Norm" from the original paper's image (Figure 1):
        # 1. Sub-layer (e.g., Multi-Head Attention)
        # 2. Add & Normalize: Add sub-layer output to input, then Layer Norm.
        # This implies: LayerNorm(x + SubLayer(x))
        # The provided `sublayer` will typically have its own dropout.
        # Let's adjust to match the paper's diagram strictly.

    def forward_post_ln(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Applies the Add & Norm operation in Post-Layer Normalization style
        (as depicted in the original Transformer paper).

        Args:
            x (torch.Tensor): The input tensor *before* the sublayer.
                              Shape: (batch_size, seq_len, d_model).
            sublayer_output (torch.Tensor): The output tensor *from* the sublayer.
                                            Shape: (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor after Add & Norm.
                          Shape: (batch_size, seq_len, d_model).
        """
        return self.norm(x + self.dropout(sublayer_output))


class EncoderLayer(nn.Module):
    """
    Implements a single Encoder Layer of the Transformer.

    Each encoder layer consists of two sub-layers:
    1. Multi-head self-attention mechanism.
    2. Position-wise fully connected feed-forward network.

    These sub-layers are each followed by a residual connection and layer normalization.

    Args:
        d_model (int): The dimension of the model's embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the inner hidden layer of the FFN.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        
        # Two AddNorm layers, one for each sub-layer
        self.add_norm1 = AddNorm(d_model, dropout_rate)
        self.add_norm2 = AddNorm(d_model, dropout_rate)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous layer or embedding.
                              Shape: (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Mask for the self-attention mechanism.
                                                Used to mask out padded positions in the source sequence.
                                                Shape: (batch_size, 1, 1, src_seq_len) or
                                                       (batch_size, 1, src_seq_len, src_seq_len).
                                                True for positions to mask.

        Returns:
            torch.Tensor: Output tensor of the encoder layer.
                          Shape: (batch_size, src_seq_len, d_model).
        """
        # Self-attention sub-layer
        # The AddNorm expects the *input* to the sublayer and the *output* of the sublayer
        attn_output = self.self_attn(x, x, x, mask=src_mask)
        x = self.add_norm1.forward_post_ln(x, attn_output)

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.add_norm2.forward_post_ln(x, ff_output)
        
        return x

class Encoder(nn.Module):
    """
    Implements the Encoder stack of the Transformer.

    The encoder consists of a stack of N identical EncoderLayers.
    It takes the source embeddings (with positional encodings) as input
    and processes them through these layers.

    Args:
        num_layers (int): The number of identical encoder layers to stack.
        d_model (int): The dimension of the model's embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the inner hidden layer of the FFN.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer norm after all encoder layers

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the encoder stack.

        Args:
            x (torch.Tensor): Input tensor (source embeddings with positional encodings).
                              Shape: (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Mask for the self-attention mechanism in encoder layers.
                                                Shape: (batch_size, 1, 1, src_seq_len).
                                                True for positions to mask.

        Returns:
            torch.Tensor: The output tensor from the encoder stack.
                          Shape: (batch_size, src_seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x) # Apply final layer norm as per original Transformer architecture

class DecoderLayer(nn.Module):
    """
    Implements a single Decoder Layer of the Transformer.

    Each decoder layer consists of three sub-layers:
    1. Masked multi-head self-attention mechanism (to prevent attending to future tokens).
    2. Multi-head attention over the output of the encoder stack.
    3. Position-wise fully connected feed-forward network.

    These sub-layers are each followed by a residual connection and layer normalization.

    Args:
        d_model (int): The dimension of the model's embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the inner hidden layer of the FFN.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.encoder_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        
        # Three AddNorm layers, one for each sub-layer
        self.add_norm1 = AddNorm(d_model, dropout_rate)
        self.add_norm2 = AddNorm(d_model, dropout_rate)
        self.add_norm3 = AddNorm(d_model, dropout_rate)

    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor], 
                memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the decoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer or target embedding.
                              Shape: (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output tensor from the encoder stack.
                                   Shape: (batch_size, src_seq_len, d_model).
            tgt_mask (Optional[torch.Tensor]): Mask for the masked self-attention mechanism in the decoder.
                                                Combines look-ahead mask and padding mask.
                                                Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len).
                                                True for positions to mask.
            memory_mask (Optional[torch.Tensor]): Mask for the encoder-decoder attention mechanism.
                                                  Used to mask out padded positions in the encoder output.
                                                  Shape: (batch_size, 1, 1, src_seq_len).
                                                  True for positions to mask.

        Returns:
            torch.Tensor: Output tensor of the decoder layer.
                          Shape: (batch_size, tgt_seq_len, d_model).
        """
        # Masked self-attention sub-layer
        # Q, K, V are all from the decoder's input (x)
        self_attn_output = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.add_norm1.forward_post_ln(x, self_attn_output)

        # Encoder-decoder attention sub-layer
        # Q is from decoder's input (x), K and V are from encoder's output (memory)
        enc_dec_attn_output = self.encoder_attn(x, memory, memory, mask=memory_mask)
        x = self.add_norm2.forward_post_ln(x, enc_dec_attn_output)

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.add_norm3.forward_post_ln(x, ff_output)

        return x

class Decoder(nn.Module):
    """
    Implements the Decoder stack of the Transformer.

    The decoder consists of a stack of N identical DecoderLayers.
    It takes the target embeddings (with positional encodings) and
    the encoder output as input, and processes them through these layers.

    Args:
        num_layers (int): The number of identical decoder layers to stack.
        d_model (int): The dimension of the model's embeddings.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the inner hidden layer of the FFN.
        dropout_rate (float): The dropout probability.
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer norm after all decoder layers

    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor], 
                memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Performs a forward pass through the decoder stack.

        Args:
            x (torch.Tensor): Input tensor (target embeddings with positional encodings).
                              Shape: (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Output tensor from the encoder stack.
                                   Shape: (batch_size, src_seq_len, d_model).
            tgt_mask (Optional[torch.Tensor]): Mask for the masked self-attention in decoder layers.
                                                Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len).
                                                True for positions to mask.
            memory_mask (Optional[torch.Tensor]): Mask for the encoder-decoder attention in decoder layers.
                                                  Shape: (batch_size, 1, 1, src_seq_len).
                                                  True for positions to mask.

        Returns:
            torch.Tensor: The output tensor from the decoder stack.
                          Shape: (batch_size, tgt_seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x) # Apply final layer norm as per original Transformer architecture

class Transformer(nn.Module):
    """
    The complete Transformer model for sequence-to-sequence tasks.

    This model comprises an encoder and a decoder, both built from
    stacked attention layers. It uses positional encodings to inject
    sequence order information and various masking strategies for
    training and inference.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        d_model (int): The dimension of the model's embeddings (and internal representations).
        num_heads (int): The number of attention heads.
        num_encoder_layers (int): The number of encoder layers to stack.
        num_decoder_layers (int): The number of decoder layers to stack.
        d_ff (int): The dimension of the inner hidden layer of the FFN.
        max_len (int): The maximum sequence length for positional encodings.
        dropout_rate (float): The dropout probability.
        pad_idx (int): The index of the padding token in the vocabulary.
                       Used to generate padding masks.
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int, 
                 num_heads: int, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 d_ff: int