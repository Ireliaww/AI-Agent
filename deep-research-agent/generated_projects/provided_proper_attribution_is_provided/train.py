To generate a complete PyTorch training script that faithfully reproduces the Transformer model's training setup as described in "Attention Is All You Need" and your analysis, we need to consider several key aspects:

1.  **Transformer Model Architecture:** Implement the Encoder-Decoder Transformer model.
2.  **Hyperparameters:** Use the specific hyperparameters detailed in the original paper for both the "base" and "big" models.
3.  **Data Loading:** Address WMT 2014 English-to-German and English-to-French. This involves BPE (Byte-Pair Encoding) tokenization, which is crucial for achieving the reported performance.
4.  **Optimizer & Learning Rate Schedule:** Implement the custom Adam optimizer with a unique learning rate schedule.
5.  **Loss Function:** Use Cross-Entropy Loss with Label Smoothing.
6.  **Training Loop:** Standard PyTorch training and validation loops.
7.  **Logging & Checkpointing:** Track progress and save model states.
8.  **Evaluation:** Calculate BLEU scores using `sacrebleu`.

**Important Considerations and Simplifications for a Self-Contained Script:**

*   **WMT 2014 Data & BPE:** Downloading and pre-processing WMT 2014 with BPE is a complex step that often involves external tools (`subword-nmt` or `sentencepiece`) and takes significant time. For a self-contained, runnable script, directly integrating `subword-nmt` or `sentencepiece` for *in-script* BPE training/application can make the script very long and require additional setup.
    *   **Approach taken:** I will provide a simplified data loading setup using `torchtext` with a basic tokenizer (e.g., `spacy`) for demonstration purposes, making the script immediately runnable.
    *   **Crucial Note:** I will provide clear instructions on how to properly integrate BPE tokenization and use the actual WMT 2014 datasets for true reproduction, as this is *essential* for achieving the reported BLEU scores. Without BPE, the vocabulary size and token representations will be vastly different, leading to much lower performance.
*   **Hardware:** Training the "big" Transformer model on WMT 2014 requires significant GPU resources (multiple GPUs are typically used, as described in the paper). The script will be set up for a single GPU by default, but can be adapted for DataParallel.
*   **Time:** Training to state-of-the-art results takes days or weeks. The provided script will run for a few epochs as a demonstration.

---

### Transformer Training Script (PyTorch)

This script provides a modular implementation of the Transformer model and its training loop, incorporating the key elements from "Attention Is All You Need."

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import time
import os
import random
import numpy as np

# For data loading (simplified for demonstration)
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.datasets import Multi30k # A smaller dataset for easier demonstration
from torchtext.data.utils import get_tokenizer

# For BLEU score calculation
from sacrebleu import corpus_bleu

# For logging
from torch.utils.tensorboard import SummaryWriter

# --- 0. Configuration and Hyperparameters ---

# Define model configuration (Base vs. Big Transformer)
class ModelConfig:
    def __init__(self, model_type="base"):
        if model_type == "base":
            self.d_model = 512       # Embedding dimension
            self.n_heads = 8         # Number of attention heads
            self.n_layers = 6        # Number of encoder/decoder layers
            self.d_ff = 2048         # Feed-forward hidden dimension
            self.dropout = 0.1       # Dropout rate
            self.warmup_steps = 4000 # Learning rate warmup steps
            self.label_smoothing = 0.1 # Label smoothing epsilon
            self.batch_size_tokens = 25000 # Approx. tokens per batch (paper uses this for big model, base often smaller but we'll try to match token count)
            self.adam_betas = (0.9, 0.98) # Adam optimizer betas
            self.adam_eps = 1e-9     # Adam optimizer epsilon
            self.max_seq_len = 256   # Max sequence length (adjust based on dataset)
        elif model_type == "big":
            self.d_model = 1024
            self.n_heads = 16
            self.n_layers = 6
            self.d_ff = 4096
            self.dropout = 0.1
            self.warmup_steps = 4000
            self.label_smoothing = 0.1
            self.batch_size_tokens = 25000 # Paper specifies this for big model
            self.adam_betas = (0.9, 0.98)
            self.adam_eps = 1e-9
            self.max_seq_len = 256
        else:
            raise ValueError("model_type must be 'base' or 'big'")

# Select model configuration
MODEL_TYPE = "base" # Change to "big" for the larger model
config = ModelConfig(MODEL_TYPE)

# General Training Configuration
NUM_EPOCHS = 10         # Number of epochs for demonstration. Real training is much longer.
CLIP = 1.0              # Gradient clipping
SAVE_DIR = "checkpoints"
LOG_DIR = "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f'transformer_{MODEL_TYPE}_{time.strftime("%Y%m%d-%H%M%S")}'))


# --- 1. Transformer Model Components ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(DEVICE)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_model)
        batch_size = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3) # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # (batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10) # Apply mask before softmax

        attention = torch.softmax(energy, dim=-1) # (batch_size, n_heads, seq_len, seq_len)
        attention = self.dropout(attention)

        x = torch.matmul(attention, V) # (batch_size, n_heads, seq_len, d_k)
        x = x.permute(0, 2, 1, 3).contiguous() # (batch_size, seq_len, n_heads, d_k)
        x = x.view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)

        x = self.fc_out(x) # (batch_size, seq_len, d_model)
        return x, attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.pos_ff = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src: (batch_size, src_seq_len, d_model)
        # src_mask: (batch_size, 1, 1, src_seq_len) or (batch_size, 1, src_seq_len, src_seq_len)

        # Self-attention
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))

        # Position-wise Feedforward
        _src = self.pos_ff(src)
        src = self.norm2(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, dropout, max_seq_len):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(DEVICE)

    def forward(self, src, src_mask):
        # src: (batch_size, src_seq_len)
        # src_mask: (batch_size, 1, 1, src_seq_len)

        src_seq_len = src.shape[1]
        
        # Token embedding + Positional Encoding
        src = self.tok_embedding(src) * self.scale
        src = self.pos_embedding(src)

        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.pos_ff = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: (batch_size, trg_seq_len, d_model)
        # enc_src: (batch_size, src_seq_len, d_model)
        # trg_mask: (batch_size, 1, trg_seq_len, trg_seq_len)
        # src_mask: (batch_size, 1, 1, src_seq_len)

        # Self-attention (masked)
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(_trg))

        # Encoder-Decoder Attention
        _trg, attention = self.enc_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout(_trg))

        # Position-wise Feedforward
        _trg = self.pos_ff(trg)
        trg = self.norm3(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_heads, d_ff, n_layers, dropout, max_seq_len):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(DEVICE)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: (batch_size, trg_seq_len)
        # enc_src: (batch_size, src_seq_len, d_model)
        # trg_mask: (batch_size, 1, trg_seq_len, trg_seq_len)
        # src_mask: (batch_size, 1, 1, src_seq_len)

        trg_seq_len = trg.shape[1]
        
        # Token embedding + Positional Encoding
        trg = self.tok_embedding(trg) * self.scale
        trg = self.pos_embedding(trg)

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg) # (batch_size, trg_seq_len, output_dim)
        return output, attention

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src: (batch_size, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, src_seq_len)
        return src_mask

    def make_trg_mask(self, trg):
        # trg: (batch_size, trg_seq_len)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, trg_seq_len)

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=DEVICE)).bool()
        # (trg_seq_len, trg_seq_len)
        
        trg_mask = trg_pad_mask & trg_sub_mask
        # (batch_size, 1, trg_seq_len, trg_seq_len)
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# --- 2. Data Loading & Preprocessing (Simplified for Demo) ---

# Define tokenizers
# For true WMT 2014 reproduction, you MUST use BPE (Byte-Pair Encoding).
# Example for BPE (requires subword-nmt or sentencepiece and pre-trained BPE model):
# from subword_nmt.apply_bpe import BPE
# bpe_codes = open('path/to/bpe_codes.txt')
# bpe = BPE(bpe_codes)
# SRC_tokenizer = lambda s: bpe.process_line(s).split()
# TRG_tokenizer = lambda s: bpe.process_line(s).split()
# For this script, we'll use Spacy tokenizer for simplicity.

spacy_en = get_tokenizer('spacy', language='en_core_web_sm')
spacy_de = get_tokenizer('spacy', language='de_core_news_sm')

# Fields for source and target languages
# We use a custom batching strategy (token-based) so batch_first=True is convenient
SRC = Field(tokenize=spacy_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=spacy_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

# Load Multi30k dataset (smaller, for demo purposes)
# For WMT 2014: you would use `torchtext.datasets.WMT14`
# WMT14.splits(exts=('.de', '.en'), fields=[SRC, TRG])
print("Loading Multi30k dataset...")
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

# Build vocabulary
print("Building vocabulary...")
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
TRG_SOS_IDX = TRG.vocab.stoi[TRG.init_token]
TRG_EOS_IDX = TRG.vocab.stoi[TRG.eos_token]

# Custom batching for token-based batch size
def batch_size_fn(new_example, current_batch, config):
    """
    Function to determine if a new example fits into the current token-based batch.
    This is a simplified version; real token-based batching is more complex.
    """
    if current_batch is None:
        return 1  # Start a new batch
    
    # Approximate token count for the new example
    new_src_len = len(new_example.src)
    new_trg_len = len(new_example.trg)
    
    # Current batch's total token count (approximate)
    current_batch_src_tokens = sum(len(ex.src) for ex in current_batch)
    current_batch_trg_tokens = sum(len(ex.trg) for ex in current_batch)
    
    # Check if adding the new example exceeds the token limit
    # We aim for ~ config.batch_size_tokens total tokens across src and trg
    if (current_batch_src_tokens + new_src_len <= config.batch_size_tokens and
        current_batch_trg_tokens + new_trg_len <= config.batch_size_tokens):
        return 1 # Add to current batch
    else:
        return config.batch_size_tokens # Create a new batch, using a dummy value indicating no more space

# Create iterators
print("Creating data iterators...")
# BucketIterator automatically groups similar length sentences to minimize padding
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=1, # Dummy batch size, actual batching by `batch_size_fn`
    sort_key=lambda x: len(x.src), # Sort by source length for efficient batching
    device=DEVICE,
    sort_within_batch=True,
    repeat=False
)

# A generator for token-based batches
def generate_token_batches(iterator, batch_size_tokens, pad_idx, device):
    batch = []
    current_src_tokens = 0
    current_trg_tokens = 0

    for example in iterator.data():
        src_len = len(example.src)
        trg_len = len(example.trg)
        
        # Check if adding this example exceeds token limit or max sequence length
        if (current_src_tokens + src_len > batch_size_tokens or
            current_trg_tokens + trg_len > batch_size_tokens or
            src_len > config.max_seq_len or trg_len > config.max_seq_len):
            
            if batch: # If there's a batch to yield
                # Pad and stack the sentences
                max_src_len = max(len(ex.src) for ex in batch)
                max_trg_len = max(len(ex.trg) for ex in batch)
                
                src_tensor = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long, device=device)
                trg_tensor = torch.full((len(batch), max_trg_len), pad_idx, dtype=torch.long, device=device)
                
                for i, ex in enumerate(batch):
                    src_tensor[i, :len(ex.src)] = torch.tensor(ex.src, dtype=torch.long, device=device)
                    trg_tensor[i, :len(ex.trg)] = torch.tensor(ex.trg, dtype=torch.long, device=device)
                
                yield src_tensor, trg_tensor

            # Reset for a new batch
            batch = [example]
            current_src_tokens = src_len
            current_trg_tokens = trg_len
        else:
            batch.append(example)
            current_src_tokens += src_len
            current_trg_tokens += trg_len
    
    if batch: # Yield any remaining batch
        max_src_len = max(len(ex.src) for ex in batch)
        max_trg_len = max(len(ex.trg) for ex in batch)
        
        src_tensor = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long, device=device)
        trg_tensor = torch.full((len(batch), max_trg_len), pad_idx, dtype=torch.long, device=device)
        
        for i, ex in enumerate(batch):
            src_tensor[i, :len(ex.src)] = torch.tensor(ex.src, dtype=torch.long, device=device)
            trg_tensor[i, :len(ex.trg)] = torch.tensor(ex.trg, dtype=torch.long, device=device)
        
        yield src_tensor, trg_tensor


# ---