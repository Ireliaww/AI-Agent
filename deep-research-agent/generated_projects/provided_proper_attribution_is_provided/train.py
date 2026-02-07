The request asks for a complete PyTorch training script to reproduce the Transformer paper's results, specifically for WMT 2014 English-to-German and English-to-French translation tasks.

This is a complex task because it requires implementing the full Transformer architecture, its specific learning rate schedule, label smoothing, and handling WMT data with BPE tokenization. The provided analysis only gives datasets, evaluation metrics, and target BLEU scores, but *not* the detailed hyperparameters or the data preparation steps.

Therefore, I will:
1.  **Implement the full Transformer architecture** as described in the original paper "Attention Is All You Need."
2.  **Use the standard hyperparameters** from the paper for both the "base" and "big" models, stating these assumptions clearly.
3.  **Provide a conceptual outline for WMT data loading and preprocessing** (BPE tokenization, vocabulary building) as these steps are typically external to the core PyTorch training script and involve downloading large datasets and using specific tools (`subword-nmt`).
4.  **For the actual training loop, I will use a simplified, dummy data generator** to allow the script to run and demonstrate the training process structure. *Users will need to replace this with their actual WMT data loaders.*
5.  **Include the custom learning rate schedule (Noam optimizer) and label smoothing loss.**
6.  **Implement a training loop with logging, checkpointing, and validation (using BLEU score).**
7.  **Structure the code modularly** for readability and maintainability.

---

### **Assumptions and Important Notes:**

1.  **Hyperparameters:** I'm using the hyperparameters (model dimensions, number of layers, dropout, optimizer settings, learning rate schedule, label smoothing) directly from the "Attention Is All You Need" paper for the "Base" and "Big" Transformer models.
2.  **Data Loading:**
    *   **Crucial Simplification:** Directly loading and preprocessing WMT 2014 data with BPE in a single, runnable Python script is highly complex and requires external tools (`subword-nmt`), large data downloads, and significant setup.
    *   **Placeholder:** The provided script includes a `data_generator` function that yields *random tensors*. This allows the script to run and demonstrate the training logic.
    *   **User Action Required:** To train on actual WMT 2014 data, you *must* replace `data_generator` with a proper data loader that:
        *   Downloads WMT 2014 En-De/En-Fr datasets.
        *   Applies Byte Pair Encoding (BPE) using a tool like `subword-nmt`.
        *   Builds a vocabulary from the BPE-encoded data.
        *   Pads sequences to the same length within batches.
        *   Yields batches of source, target, and target output (shifted right) tensors.
    *   **Recommended Data Loading Libraries:** For real WMT data, consider using libraries like `torchtext` (though it can be tricky with BPE) or implementing a custom `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` after preprocessing the data with `subword-nmt`.
3.  **BLEU Score Calculation:** The script includes `sacrebleu` for evaluation. When using real data, ensure your generated translations are properly detokenized before calculating BLEU. The `evaluate_bleu` function is a placeholder that would need real data and model inference.
4.  **Multi-GPU Training:** The original paper used 8 P100 GPUs. This script defaults to single GPU. For multi-GPU, you would typically wrap the model with `nn.DataParallel(model)` (simpler) or `nn.parallel.DistributedDataParallel(model)` (more performant for large scale) and configure your data loaders for distributed training.
5.  **Training Duration:** The paper mentions 3.5 days on 8 P100 GPUs for the big model. The provided script will run for a predefined number of steps (epochs for dummy data) and will not reach the paper's performance without the correct data and hardware.
6.  **`tqdm`:** Added for progress bars during training. Install with `pip install tqdm`.
7.  **`sacrebleu`:** Added for BLEU calculation. Install with `pip install sacrebleu`.

---

### **PyTorch Training Script for Transformer**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import time
import os
from tqdm import tqdm
import sacrebleu # For BLEU score calculation

# --- 1. Transformer Model Architecture (from "Attention Is All You Need" paper) ---

# Helper function for cloning modules
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# --- Core Attention Mechanism ---
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # Dimension of K, Q, V for each head
        self.h = h               # Number of attention heads
        self.linears = clones(nn.Linear(d_model, d_model), 4) # W_Q, W_K, W_V, W_O
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # Same mask applied to all h heads
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat and apply final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill with a very small number for masked positions
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

# --- Position-wise Feed-Forward Networks ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

# --- Layer Normalization ---
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# --- Sublayer Connection (Residual + LayerNorm + Dropout) ---
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Apply layer normalization, then sublayer, then dropout, then add residual
        return x + self.dropout(sublayer(self.norm(x)))

# --- Encoder Layer ---
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # Two sublayers
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# --- Decoder Layer ---
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn    # Masked self-attention
        self.src_attn = src_attn      # Encoder-decoder attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3) # Three sublayers

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# --- Embeddings and Positional Encoding ---
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # Not a learnable parameter

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

# --- Full Transformer Model ---
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# --- Output Generator (Linear + Softmax) ---
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x) # Log softmax is usually applied in the loss function

# --- 2. Model Initialization Function ---
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important in the paper for stability. Initialize parameters with Glorot/Xavier.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# --- 3. Custom Learning Rate Schedule (Noam Optimizer) ---
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

# --- 4. Label Smoothing Loss ---
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.requires_grad_(False))

# --- 5. Data Batching and Masking ---
class Batch:
    def __init__(self, src, tgt=None, pad_idx=0):
        self.src = src
        self.src_mask = (src != pad_idx).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1] # Target input (shifted right)
            self.tgt_y = tgt[:, 1:] # Target output (what we want to predict)
            self.tgt_mask = \
                self.make_std_mask(self.tgt, pad_idx)
            self.ntokens = (self.tgt_y != pad_idx).sum().item()

    @staticmethod
    def make_std_mask(tgt, pad_idx):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad_idx).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    ) == 0
    return subsequent_mask

# --- 6. Data Loading (Conceptual / Placeholder) ---

# This is a placeholder for actual WMT data loading.
# In a real scenario, you would:
# 1. Download WMT 2014 En-De or En-Fr data.
# 2. Train a BPE model and apply it to the data using `subword-nmt`.
# 3. Build a vocabulary mapping tokens to indices.
# 4. Create a PyTorch Dataset and DataLoader.
# For simplicity, this function generates random data.
def data_generator(src_vocab_size, tgt_vocab_size, batch_size, num_batches, max_len=50):
    for i in range(num_batches):
        src = torch.randint(1, src_vocab_size, (batch_size, max_len))
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, max_len))
        # Ensure start and end tokens, and some padding
        src[:, 0] = 1 # BOS
        src[:, -1] = 2 # EOS
        tgt[:, 0] = 1 # BOS
        tgt[:, -1] = 2 # EOS
        # Randomly mask some tokens as padding (index 0)
        padding_mask_src = torch.rand(batch_size, max_len) < 0.1
        padding_mask_tgt = torch.rand(batch_size, max_len) < 0.1
        src[padding_mask_src] = 0
        tgt[padding_mask_tgt] = 0

        # Create a Batch object
        yield Batch(src, tgt, pad_idx=0)


# --- 7. Training and Evaluation Functions ---

def run_epoch(data_iter, model, loss_compute, optimizer, device, is_train=True):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    mode_str = "Train" if is_train else "Valid"
    pbar = tqdm(data_iter, desc=f"{mode_str} Epoch", leave=False)

    for i, batch in enumerate(pbar):
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        src_mask = batch.src_mask.to(device)
        tgt_mask = batch.tgt_mask.to(device)
        tgt_y = batch.tgt_y.to(device)
        ntokens = batch.ntokens

        out = model.forward(src, tgt, src_mask, tgt_mask)
        # Apply log_softmax for KLDivLoss
        prob = model.generator(out).log_softmax(dim=-1)
        # Reshape for loss calculation: (batch_size * sequence_length, vocab_size)
        # and target: (batch_size * sequence_length)
        loss = loss_compute(prob.contiguous().view(-1, prob.size(-1)),
                            tgt_y.contiguous().view(-1))
        
        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.optimizer.zero_grad() # NoamOpt wraps the base optimizer

        total_loss += loss.item()
        total_tokens += ntokens
        tokens += ntokens
        
        pbar.set_postfix({'loss': loss.item() / ntokens, 'tokens/sec': tokens / (time.time() - start)})

    elapsed = time.time() - start
    return total_loss / total_tokens, elapsed

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           ys,
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data))).to(device)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)
        if next_word == end_symbol:
            break
    return ys

def evaluate_bleu(model, data_iter, src_vocab, tgt_vocab, device, max_len=50, start_symbol=1, end_symbol=2):
    model.eval()
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iter, desc="Evaluating BLEU", leave=False)):
            src = batch.src.to(device)
            src_mask = batch.src_mask.to(device)
            
            # Decode one sentence at a time for simplicity in greedy_decode
            for j in range(src.size(0)):
                single_src = src[j:j+1]
                single_src_mask = src_mask[j:j+1]
                
                out_tokens = greedy_decode(model, single_src, single_src_mask, max_len, start_symbol, end_symbol, device)
                
                # Convert token IDs to words (dummy conversion for now)
                # In a real scenario, you'd use your actual vocabulary to map IDs to BPE tokens
                # and then detokenize the BPE tokens.
                predicted_sentence = " ".join([str(x.item()) for x in out_tokens[0] if x.item() not in [0, start_symbol, end_symbol]])
                target_sentence = " ".join([str(x.item()) for x in batch.tgt_y[j] if x.item() not in [0, start_symbol, end_symbol]])
                
                hypotheses.append(predicted_sentence)
                references.append([target_sentence]) # sacrebleu expects a list of references

    if not hypotheses:
        return 0.0 # No predictions made

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return bleu.score

# --- Main Training Script ---
def main():
    # --- Configuration ---
    # Choose between 'base' and 'big' model configurations
    model_type = "base" # Or "big"
    task = "en-de"      # Or "en-fr"

    if model_type == "base":
        N_layers = 6
        d_model = 512
        d_ff = 2048
        h_heads = 8
        dropout = 0.1
        batch_size = 25000 # tokens per batch (approximate for dummy data)
        warmup_steps = 4000
    elif model_type == "big":
        N_layers = 6
        d_model = 1024
        d_ff = 4096
        h_heads = 16
        dropout = 0.3
        batch_size = 25000 # tokens per batch (approximate for dummy data)
        warmup_steps = 4000
    else:
        raise ValueError("model_type must be 'base' or 'big'")

    # Common parameters
    src_vocab_size = 10000 # Placeholder: actual vocab size will be larger (e.g.,