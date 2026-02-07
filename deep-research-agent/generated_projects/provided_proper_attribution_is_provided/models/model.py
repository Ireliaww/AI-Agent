Here's a complete PyTorch implementation of the Transformer model as described in "Attention Is All You Need" (Vaswani et al., 2017), adhering to your requirements.

This implementation breaks down the Transformer into its core components, making it modular and easy to understand.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in the paper.
    This module injects information about the relative or absolute position
    of tokens in the sequence, as the Transformer does not use recurrence
    or convolution.

    The positional encoding is added to the input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model's embeddings (e.g., 512).
            max_len (int): The maximum sequence length for which to generate
                           positional encodings. This pre-computes the encodings.
            dropout (float): Dropout rate to apply to the output of positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe) # Register as a buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added, of shape
                          (batch_size, seq_len, d_model).
        """
        # x has shape (batch_size, seq_len, d_model)
        # self.pe has shape (1, max_len, d_model)
        # We take the first 'seq_len' positional encodings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

    This module handles the core attention mechanism, including optional masking.
    """
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculates scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len_q, d_k).
            k (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len_k, d_k).
            v (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_len_v, d_v).
                              Note: seq_len_k must be equal to seq_len_v.
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
                                           or (1, 1, seq_len_q, seq_len_k).
                                           Masked positions will be set to a very small negative
                                           number before softmax, effectively ignoring them.
                                           Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_heads, seq_len_q, d_v).
        """
        d_k = q.size(-1)
        
        # (batch_size, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Masking: wherever mask is True (or 1), set scores to a very small negative value.
            # This ensures that after softmax, these positions will have probabilities close to 0.
            scores = scores.masked_fill(mask == 0, -1e9) 
            # Note: The original paper uses mask==0 to indicate positions to be masked out (e.g., padding).
            # For look-ahead mask, it's typically 0 for positions to be ignored.

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v) # (batch_size, n_heads, seq_len_q, d_v)
        return output

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention mechanism.
    It projects the queries, keys, and values h times with different, learned
    linear projections to d_k, d_k, and d_v dimensions, respectively.
    On each of these projected versions of queries, keys, and values,
    attention function is applied in parallel, yielding h output values.
    These are concatenated and once again projected, resulting in the final values.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): The dimension of the model's embeddings (e.g., 512).
            n_heads (int): The number of attention heads (e.g., 8).
            dropout (float): Dropout rate to apply to the attention weights.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads # Dimension of K and Q for each head
        self.d_v = d_model // n_heads # Dimension of V for each head
        self.n_heads = n_heads
        self.d_model = d_model

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final linear projection for the concatenated outputs
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs multi-head attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
            mask (torch.Tensor, optional): Mask tensor for attention scores.
                                           Shape (batch_size, 1, seq_len_q, seq_len_k)
                                           or (1, 1, seq_len_q, seq_len_k). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        batch_size = q.size(0)

        # 1. Apply linear projections and reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k/d_v)
        # -> (batch_size, n_heads, seq_len, d_k/d_v)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 2. Apply scaled dot-product attention
        # (batch_size, n_heads, seq_len_q, d_v)
        attn_output = self.attention(q, k, v, mask=mask)

        # 3. Concatenate heads and apply final linear projection
        # (batch_size, n_heads, seq_len_q, d_v) -> (batch_size, seq_len_q, n_heads, d_v)
        # -> (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(attn_output) # (batch_size, seq_len_q, d_model)
        return self.dropout(output)

class PositionWiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network (FFN).
    This consists of two linear transformations with a ReLU activation in between.
    It is applied to each position separately and identically.

    FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes the PositionWiseFeedForward module.

        Args:
            d_model (int): The dimension of the model's embeddings (e.g., 512).
            d_ff (int): The dimension of the inner layer of the FFN (e.g., 2048).
            dropout (float): Dropout rate to apply after the second linear layer.
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feed-forward network to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Encoder.
    It consists of two sub-layers:
    1. Multi-head self-attention mechanism.
    2. Position-wise fully connected feed-forward network.

    Each sub-layer is followed by a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes an EncoderLayer.

        Args:
            d_model (int): The dimension of the model's embeddings.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner layer of the FFN.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous layer,
                              shape (batch_size, src_seq_len, d_model).
            src_mask (torch.Tensor): Source mask for self-attention,
                                     shape (batch_size, 1, 1, src_seq_len) or 
                                     (batch_size, 1, src_seq_len, src_seq_len).
                                     Used to mask out padding tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, src_seq_len, d_model).
        """
        # Sub-layer 1: Multi-head self-attention
        # Apply LayerNorm before attention (pre-norm)
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=src_mask)
        x = x + self.dropout1(attn_output) # Add residual connection and dropout

        # Sub-layer 2: Position-wise feed-forward network
        # Apply LayerNorm before FFN (pre-norm)
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output) # Add residual connection and dropout
        
        return x

class Encoder(nn.Module):
    """
    The Transformer Encoder, composed of a stack of N identical EncoderLayers.
    It takes source embeddings and positional encodings as input.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, 
                 n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the Encoder.

        Args:
            vocab_size (int): Size of the source vocabulary.
            d_model (int): The dimension of the model's embeddings.
            n_layers (int): The number of encoder layers.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner layer of the FFN.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer normalization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Encoder.

        Args:
            src (torch.Tensor): Source input tensor of token IDs,
                                shape (batch_size, src_seq_len).
            src_mask (torch.Tensor): Source mask for self-attention,
                                     shape (batch_size, 1, 1, src_seq_len) or 
                                     (batch_size, 1, src_seq_len, src_seq_len).
                                     Used to mask out padding tokens.

        Returns:
            torch.Tensor: Output tensor from the last encoder layer,
                          shape (batch_size, src_seq_len, d_model).
        """
        # 1. Input Embedding + Positional Encoding
        x = self.embedding(src) * math.sqrt(self.norm.normalized_shape[0]) # Scale embeddings
        x = self.pos_encoder(x)
        x = self.dropout(x) # Apply dropout after embedding and PE

        # 2. Pass through N encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        # 3. Final LayerNorm (as per original paper's implementation details)
        x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    """
    Represents a single layer of the Transformer Decoder.
    It consists of three sub-layers:
    1. Masked multi-head self-attention mechanism (on target sequence).
    2. Multi-head attention mechanism over the encoder's output.
    3. Position-wise fully connected feed-forward network.

    Each sub-layer is followed by a residual connection and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a DecoderLayer.

        Args:
            d_model (int): The dimension of the model's embeddings.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner layer of the FFN.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                tgt_mask: torch.Tensor, src_tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the decoder layer.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer,
                              shape (batch_size, tgt_seq_len, d_model).
            encoder_output (torch.Tensor): Output from the encoder,
                                           shape (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor): Mask for decoder self-attention (look-ahead mask combined
                                     with padding mask),
                                     shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
            src_tgt_mask (torch.Tensor): Mask for encoder-decoder attention (padding mask for
                                         encoder output),
                                         shape (batch_size, 1, 1, src_seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        # Sub-layer 1: Masked Multi-head self-attention
        norm_x = self.norm1(x)
        attn1 = self.self_attn(norm_x, norm_x, norm_x, mask=tgt_mask)
        x = x + self.dropout1(attn1)

        # Sub-layer 2: Multi-head encoder-decoder attention
        # Query from decoder, Key and Value from encoder output
        norm_x = self.norm2(x)
        attn2 = self.encoder_attn(norm_x, encoder_output, encoder_output, mask=src_tgt_mask)
        x = x + self.dropout2(attn2)

        # Sub-layer 3: Position-wise feed-forward network
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout3(ff_output)
        
        return x

class Decoder(nn.Module):
    """
    The Transformer Decoder, composed of a stack of N identical DecoderLayers.
    It takes target embeddings, positional encodings, and encoder output as input.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, 
                 n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the Decoder.

        Args:
            vocab_size (int): Size of the target vocabulary.
            d_model (int): The dimension of the model's embeddings.
            n_layers (int): The number of decoder layers.
            n_heads (int): The number of attention heads.
            d_ff (int): The dimension of the inner layer of the FFN.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model) # Final layer normalization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor, 
                tgt_mask: torch.Tensor, src_tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Decoder.

        Args:
            tgt (torch.Tensor): Target input tensor of token IDs,
                                shape (batch_size, tgt_seq_len).
            encoder_output (torch.Tensor): Output from the encoder,
                                           shape (batch_size, src_seq_len, d_model).
            tgt_mask (torch.Tensor): Mask for decoder self-attention (look-ahead mask combined
                                     with padding mask),
                                     shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
            src_tgt_mask (torch.Tensor): Mask for encoder-decoder attention (padding mask for
                                         encoder output),
                                         shape (batch_size, 1, 1, src_seq_len).

        Returns:
            torch.Tensor: Output tensor from the last decoder layer,
                          shape (batch_size, tgt_seq_len, d_model).
        """
        # 1. Input Embedding + Positional Encoding
        x = self.embedding(tgt) * math.sqrt(self.norm.normalized_shape[0]) # Scale embeddings
        x = self.pos_encoder(x)
        x = self.dropout(x) # Apply dropout after embedding and PE

        # 2. Pass through N decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_tgt_mask)
        
        # 3. Final LayerNorm (as per original paper's implementation details)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    """
    The complete Transformer model, an encoder-decoder architecture
    based entirely on attention mechanisms.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, 
                 n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, max_len: int = 5000, pad_idx: int = 0):
        """
        Initializes the Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            d_model (int): The dimension of the model's embeddings (default: 512).
            n_layers (int): The number of encoder and decoder layers (default: 6).
            n_heads (int): The number of attention heads (default: 8).
            d_ff (int): The dimension of the inner layer of the FFN (default: 2048).
            dropout (float): Dropout rate (default: 0.1).
            max_len (int): Maximum sequence length for positional encoding (default: 5000).
            pad_idx (int): Index of the padding token in the vocabulary.
                           Used to create masks. (default: 0).
        """
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.generator = nn.Linear(d_model, tgt_vocab_size) # Output layer for token probabilities
        self.pad_idx = pad_idx

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates a padding mask for the source sequence.
        This mask prevents attention to padding tokens.

        Args:
            src (torch.Tensor): Source input tensor of token IDs,
                                shape (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Source padding mask,
                          shape (batch_size, 1, 1, src_seq_len).
                          True where it's a real token, False where it's padding.
        """
        # (batch_size, 1, 1, src_seq_len)
        # 1 if not pad_idx, 0 if pad_idx
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Creates a combined look-ahead mask and padding mask for the target sequence.
        The look-ahead mask prevents attending to future tokens during decoding.
        The padding mask prevents attention to padding tokens.

        Args:
            tgt (torch.Tensor): Target input tensor of token IDs,
                                shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Target attention mask,
                          shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
        """
        tgt_seq_len = tgt.size(1)

        # Look-ahead mask (upper triangular matrix with 0s on diagonal and below)
        # (tgt_seq_len, tgt_seq_len)
        look_ahead_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).bool().to(tgt.device)
        
        # Padding mask for target sequence
        # (batch_size, 1, 1, tgt_seq_len) -> (batch_size, 1, tgt_seq_len, 1) after transpose
        # -> (batch_size, 1, tgt_seq_len, tgt_seq_len) after broadcasting
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2).bool()

        # Combine masks: element-wise AND
        # look_ahead_mask is (tgt_seq_len, tgt_seq_len), tgt_pad_mask is (batch_size, 1, 1, tgt_seq_len)
        # Result will be (batch_size, 1, tgt_seq_len, tgt_seq_len) due to broadcasting
        tgt_mask = look_ahead_mask & tgt_pad_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the entire Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of token IDs,
                                shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target input tensor of token IDs (shifted right),
                                shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Output logits for the target vocabulary,
                          shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        # 1. Create masks
        src_mask = self._make_src_mask(src) # For encoder self-attention and encoder-decoder attention K, V
        tgt_mask = self._make_tgt_mask(tgt) # For decoder self-attention Q, K, V

        # The src_tgt_mask is used in the decoder's encoder-decoder attention.
        # It masks the *encoder output* (keys and values) based on padding in the *source*.
        # So, it's essentially the src_mask, but broadcasted correctly for the decoder's cross-attention.
        # Its shape needs to be (batch_size, 1, tgt_seq_len, src_seq_len)
        # (src_mask is (batch_size, 1, 1, src_seq_len), which broadcasts correctly against Q of shape (batch, 1, tgt_len, 1))
        src_tgt_mask = src_mask 

        # 2. Encode source sequence
        encoder_output = self.encoder(src, src_mask)

        # 3.