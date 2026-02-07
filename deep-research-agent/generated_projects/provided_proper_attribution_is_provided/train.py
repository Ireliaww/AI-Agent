The following PyTorch training script provides a comprehensive implementation of the Transformer model, designed to reproduce the core experimental setup and results described in the "Attention Is All You Need" paper.

**Important Considerations and Simplifications:**

1.  **WMT Data Loading:** The WMT 2014 English-to-German and English-to-French datasets are large and require extensive preprocessing (tokenization, byte-pair encoding (BPE), vocabulary building, etc.) that is beyond the scope of a single, self-contained Python script. This script uses a *conceptual* `torchtext` setup for demonstration. In a real-world scenario, you would:
    *   Download the raw WMT data.
    *   Use tools like `subword-nmt` or `sentencepiece` for BPE.
    *   Build vocabularies from the BPE-encoded data.
    *   Create `torchtext.data.Field` objects with these pre-built vocabularies and appropriate tokenization.
    *   Load the preprocessed data using `torchtext.data.TabularDataset` or similar.
    *   **For a runnable script without manual preprocessing, you might consider using a smaller, pre-processed translation dataset or mock data.** The current data loading part serves as a placeholder to illustrate the structure.
2.  **BLEU Score Calculation:** Calculating BLEU scores during training is computationally intensive and often requires specific post-processing (detokenization). The script focuses on validation loss. For actual BLEU evaluation, you would typically:
    *   Generate translations for the validation/test set using the trained model.
    *   Detokenize the generated translations.
    *   Use a standard BLEU evaluation script (e.g., `multi-bleu.perl` from Moses) to compare with reference translations.
3.  **Token-based Batching:** The paper uses batching based on a maximum number of *tokens* per batch (e.g., 25000 tokens), which implies dynamic batch sizing. This script uses a fixed number of *sequences* per batch for simplicity. Implementing dynamic token-based batching with PyTorch `DataLoader` requires custom collate functions and careful handling.
4.  **Multi-GPU Training:** The paper used 8 P100 GPUs. This script is designed for a single GPU. For multi-GPU training, you would wrap the model with `torch.nn.DataParallel` or `torch.nn.DistributedDataParallel`.
5.  **Hyperparameters:** The hyperparameters are set to the "Base Model" configuration as described in the paper, with notes on the "Big Model" differences.

---

### PyTorch Transformer Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- 1. Hyperparameters ---
class HParams:
    # Model Architecture (Base Model)
    d_model = 512  # Dimension of model embeddings
    n_head = 8     # Number of attention heads
    n_layers = 6   # Number of encoder and decoder layers
    d_ff = 2048    # Dimension of the feed-forward network
    dropout = 0.1  # Dropout rate

    # Training
    batch_size = 64 # Number of sequences per batch (simplified from token-based)
    epochs = 100   # Number of training epochs
    warmup_steps = 4000 # Learning rate warmup steps
    label_smoothing = 0.1 # Label smoothing epsilon

    # Optimizer (Adam with specific betas and epsilon)
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9

    # Data
    max_seq_len = 256 # Maximum sequence length for training
    # For WMT-14 En-De: vocab sizes typically 30k-40k for BPE
    # For WMT-14 En-Fr: vocab sizes typically 30k-40k for BPE
    # These will be determined by the actual vocabulary built from data
    src_vocab_size = None # Placeholder
    tgt_vocab_size = None # Placeholder

    # Checkpointing & Logging
    log_interval = 100 # Log every N batches
    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Big Model differences (for reference)
    # d_model = 1024
    # n_head = 16
    # d_ff = 4096

# --- 2. Model Components ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # Add head dimension
        
        n_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => n_head x d_k
        query, key, value = \
            [l(x).view(n_batches, -1, self.n_head, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_head * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill with a very small number
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

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

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_head, n_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        c = HParams
        attn = MultiHeadAttention(d_model, n_head, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), n_layers)
        self.decoder = Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), n_layers)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), position)
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), position)
        self.generator = Generator(d_model, tgt_vocab)

        # This was important from the paper, but we didn't implement it in the book.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# --- 3. Utility Functions ---

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0
    return subsequent_mask.to(HParams.device)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.model_size**(-0.5) * min(step**(-0.5), step * self.warmup**(-1.5))

class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."
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
        true_dist.fill_(self.smoothing / (self.size - 2)) # -2 for <unk> and <pad>
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.requires_grad_(False))

# --- 4. Data Loading & Preprocessing (Conceptual/Simplified) ---

# For a real WMT setup, you'd use torchtext.data.Field and torchtext.data.TabularDataset
# and handle tokenization (e.g., BPE), vocabulary building, etc.

# Mock data generation for demonstration
def generate_mock_data(src_vocab_size, tgt_vocab_size, num_samples, max_len):
    src_data = torch.randint(2, src_vocab_size, (num_samples, max_len)).long()
    tgt_data = torch.randint(2, tgt_vocab_size, (num_samples, max_len)).long()
    # Replace some tokens with padding_idx (0) to simulate varying lengths
    for i in range(num_samples):
        src_len = torch.randint(10, max_len, (1,)).item()
        tgt_len = torch.randint(10, max_len, (1,)).item()
        if src_len < max_len:
            src_data[i, src_len:] = 0 # PAD_IDX
        if tgt_len < max_len:
            tgt_data[i, tgt_len:] = 0 # PAD_IDX
    return TensorDataset(src_data, tgt_data)

def load_data(hparams):
    # In a real scenario, these would be loaded from pre-processed WMT files
    # For demonstration, we'll use mock data and set vocab sizes
    print("Generating mock data... (Replace with actual WMT data loading)")
    
    # Mock vocab sizes (e.g., 30k for common BPE vocab)
    hparams.src_vocab_size = 30000
    hparams.tgt_vocab_size = 30000
    
    train_dataset = generate_mock_data(hparams.src_vocab_size, hparams.tgt_vocab_size, 10000, hparams.max_seq_len)
    val_dataset = generate_mock_data(hparams.src_vocab_size, hparams.tgt_vocab_size, 1000, hparams.max_seq_len)

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)
    
    # Define special tokens (PAD_IDX is crucial for masking and loss)
    # Typically, PAD_IDX=0, BOS_IDX=1, EOS_IDX=2, UNK_IDX=3
    PAD_IDX = 0
    BOS_IDX = 1 # Not explicitly used for input in this script but common
    EOS_IDX = 2 # Not explicitly used for input in this script but common

    print(f"Source Vocabulary Size: {hparams.src_vocab_size}")
    print(f"Target Vocabulary Size: {hparams.tgt_vocab_size}")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")

    return train_dataloader, val_dataloader, PAD_IDX

# --- 5. Training and Evaluation Functions ---

def make_masks(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(-2)
    tgt_mask = (tgt != pad_idx).unsqueeze(-2) & subsequent_mask(tgt.size(-1)).type_as(tgt.data)
    return src_mask, tgt_mask

def run_epoch(dataloader, model, loss_compute, optimizer=None, is_train=True):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(dataloader):
        src, tgt = batch[0].to(HParams.device), batch[1].to(HParams.device)
        
        # Shift target for input (remove last token) and output (remove first token)
        # The target input to the decoder is the target sequence shifted right by one.
        # So, if original target is [BOS, w1, w2, EOS, PAD], decoder input is [BOS, w1, w2, EOS]
        # And the labels for loss calculation are [w1, w2, EOS, PAD]
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]

        src_mask, tgt_mask = make_masks(src, tgt_input, loss_compute.padding_idx)

        out = model(src, tgt_input, src_mask, tgt_mask)
        
        # Flatten for loss calculation
        # out: (batch_size * seq_len, vocab_size)
        # tgt_labels: (batch_size * seq_len)
        loss = loss_compute(out.contiguous().view(-1, out.size(-1)), tgt_labels.contiguous().view(-1))
        
        num_tokens = (tgt_labels != loss_compute.padding_idx).sum().item()
        total_loss += loss.item()
        total_tokens += num_tokens
        tokens += num_tokens

        if is_train:
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % HParams.log_interval == 0:
                elapsed = time.time() - start
                print(f"Epoch Step: {i}/{len(dataloader)} | Loss: {loss.item() / num_tokens:.3f} | "
                      f"Tokens/sec: {tokens / elapsed:.2f} | Elapsed: {elapsed:.2f}s")
                start = time.time()
                tokens = 0

    return total_loss / total_tokens

# --- 6. Main Training Loop ---

def train():
    os.makedirs(HParams.checkpoint_dir, exist_ok=True)

    train_dataloader, val_dataloader, PAD_IDX = load_data(HParams)

    # Initialize model
    model = Transformer(
        HParams.src_vocab_size, HParams.tgt_vocab_size,
        HParams.d_model, HParams.n_head, HParams.n_layers, HParams.d_ff, HParams.dropout
    ).to(HParams.device)

    # Optimizer
    # Adam with specific parameters as per paper
    optimizer = NoamOpt(HParams.d_model, HParams.warmup_steps,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(HParams.beta1, HParams.beta2), eps=HParams.eps))

    # Loss criterion
    criterion = LabelSmoothingLoss(size=HParams.tgt_vocab_size, padding_idx=PAD_IDX, smoothing=HParams.label_smoothing)

    best_val_loss = float('inf')

    print(f"Starting training on {HParams.device}...")
    for epoch in range(HParams.epochs):
        model.train()
        print(f"\nEpoch {epoch + 1}/{HParams.epochs} (Training)")
        train_loss = run_epoch(train_dataloader, model, criterion, optimizer, is_train=True)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

        model.eval()
        print(f"Epoch {epoch + 1} (Validation)")
        with torch.no_grad():
            val_loss = run_epoch(val_dataloader, model, criterion, is_train=False)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), HParams.best_model_path)
        else:
            print(f"Validation loss did not improve. Current best: {best_val_loss:.4f}")
        
        # Save a checkpoint for the current epoch (optional, for recovery)
        current_epoch_path = os.path.join(HParams.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), current_epoch_path)
        print(f"Saved model checkpoint for epoch {epoch+1} to {current_epoch_path}")

    print("\nTraining complete!")
    print(f"Best model saved to {HParams.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument('--epochs', type=int, default=HParams.epochs,
                        help=f'Number of training epochs (default: {HParams.epochs})')
    parser.add_argument('--batch_size', type=int, default=HParams.batch_size,
                        help=f'Batch size (default: {HParams.batch_size})')
    parser.add_argument('--d_model', type=int, default=HParams.d_model,
                        help=f'Model dimension (default: {HParams.d_model})')
    parser.add_argument('--n_head', type=int, default=HParams.n_head,
                        help=f'Number of attention heads (default: {HParams.n_head})')
    parser.add_argument('--n_layers', type=int, default=HParams.n_layers,
                        help=f'Number of encoder/decoder layers (default: {HParams.n_layers})')
    parser.add_argument('--d_ff', type=int, default=HParams.d_ff,
                        help=f'Feed-forward dimension (default: {HParams.d_ff})')
    parser.add_argument('--dropout', type=float, default=HParams.dropout,
                        help=f'Dropout rate (default: {HParams.dropout})')
    parser.add_argument('--warmup_steps', type=int, default=HParams.warmup_steps,
                        help=f'Learning rate warmup steps (default: {HParams.warmup_steps})')
    parser.add_argument('--label_smoothing', type=float, default=HParams.label_smoothing,
                        help=f'Label smoothing epsilon (default: {HParams.label_smoothing})')
    parser.add_argument('--log_interval', type=int, default=HParams.log_interval,
                        help=f'Log interval (default: {HParams.log_interval})')
    
    args = parser.parse_