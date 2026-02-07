The request asks for a complete PyTorch training script for the Transformer model, based on the provided analysis of the "Attention Is All You Need" paper. While the analysis outlines the experimental setup and key results, it *does not provide the architectural details or hyperparameters* necessary to implement the model and its training from scratch.

To fulfill this request accurately, I will draw upon the well-known details from the original paper, "Attention Is All You Need" (Vaswani et al., 2017), to reconstruct the model architecture, hyperparameters, and training methodology.

**Key Information Sourced from the Original Paper (Vaswani et al., 2017):**

1.  **Model Architecture (Base Model):**
    *   `d_model` (embedding dimension): 512
    *   `num_heads` (attention heads): 8
    *   `d_ff` (feed-forward inner dimension): 2048
    *   `num_layers` (encoder/decoder layers): 6
    *   Dropout: 0.1 (applied to sub-layer outputs before adding to input, and to embeddings + positional encodings)

2.  **Hyperparameters:**
    *   **Optimizer:** Adam (`beta1=0.9`, `beta2=0.98`, `eps=1e-9`)
    *   **Learning Rate Schedule:** Custom schedule: `d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)`
        *   `warmup_steps`: 4000
    *   **Batch Size:** Approximately 25,000 source tokens and 25,000 target tokens per batch. (For practical demonstration on a single GPU, we will use a smaller, sentence-based batch size, acknowledging this deviation).
    *   **Label Smoothing:** `epsilon_ls = 0.1`
    *   **Weight Initialization:** Xavier initialization for linear layers.
    *   **Training Steps:** 300,000 steps for En-De, 100,000 steps for En-Fr (base model).

3.  **Data Preprocessing:**
    *   Byte-Pair Encoding (BPE) for subword units.
    *   Shared vocabulary between source and target for En-De, separate for En-Fr.
    *   Maximum sequence length: 256 tokens.

4.  **Evaluation:**
    *   BLEU score using `multi-bleu.perl` (or `sacreBLEU` for modern reproducibility).

---

### **PyTorch Training Script for Transformer (Base Model)**

This script will provide a comprehensive implementation of the Transformer model and its training loop based on the "Attention Is All You Need" paper.

**Important Considerations and Limitations:**

*   **Data Handling:** Actual WMT 2014 data requires specific download, tokenization (e.g., Moses tokenizer), and BPE preprocessing (e.g., `subword-nmt`). For simplicity and demonstration, this script will use `torchtext` with `spacy` for basic tokenization and a simplified `SubwordField` for BPE-like functionality. To achieve the paper's results, you would need the exact WMT preprocessing pipeline.
*   **Computational Resources:** Training a full Transformer on WMT datasets to achieve state-of-the-art results requires significant computational power (multiple GPUs, days/weeks of training). This script is a *framework* demonstrating the components and training process; it will not achieve SOTA results on a typical single GPU within a reasonable demonstration timeframe.
*   **Batch Size:** The paper used a token-based batch size of 25,000. Implementing this efficiently often requires gradient accumulation or distributed training. For a self-contained script, we'll use a smaller, sentence-based batch size for feasibility, but note the original paper's approach.
*   **BLEU Score:** BLEU calculation during training will be done on generated sentences. For robust evaluation, it's typically done on a full, detokenized test set.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import time
import os
import random
from torch.utils.tensorboard import SummaryWriter

# For data loading and preprocessing
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.vocab import Vocab
from collections import Counter
import subword_nmt.learn_bpe as learn_bpe
import subword_nmt.apply_bpe as apply_bpe
import io

# For BLEU score calculation
from sacrebleu.metrics import BLEU

# --- 0. Configuration and Hyperparameters ---
class Config:
    # Model Hyperparameters (Base Model from paper)
    d_model = 512       # Embedding dimension
    n_heads = 8         # Number of attention heads
    d_ff = 2048         # Feed-forward hidden dimension
    n_layers = 6        # Number of encoder/decoder layers
    dropout_rate = 0.1  # Dropout rate
    max_seq_len = 256   # Maximum sequence length

    # Training Hyperparameters
    batch_size = 64     # Simplified batch size (paper used ~25k tokens)
    epochs = 10         # Number of epochs for demonstration (paper used 100k-600k steps)
    lr_warmup_steps = 4000 # Learning rate warmup steps
    label_smoothing = 0.1 # Label smoothing epsilon
    
    # Optimizer Hyperparameters (from paper)
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-9

    # Data paths and setup
    data_dir = './data'
    src_lang = 'en'
    trg_lang = 'de'
    
    # BPE parameters
    bpe_codes_path = os.path.join(data_dir, f'bpe_codes.{src_lang}-{trg_lang}')
    bpe_vocab_size = 10000 # Example BPE vocab size

    # Checkpointing and Logging
    save_dir = './checkpoints'
    log_dir = './logs'
    checkpoint_interval = 1 # Save every X epochs
    log_interval = 100 # Log every X steps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories if they don't exist
os.makedirs(Config.data_dir, exist_ok=True)
os.makedirs(Config.save_dir, exist_ok=True)
os.makedirs(Config.log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(Config.log_dir)

# --- 1. Data Loading and Preprocessing ---
# Helper for BPE tokenization
class BPEField(Field):
    def __init__(self, codes_path, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.codes_path = codes_path
        self.vocab_size = vocab_size
        self.bpe = None # Will be initialized after fitting BPE

    def build_vocab(self, *args, **kwargs):
        # This method is usually called by torchtext.Dataset.build_vocab
        # We override it to apply BPE first then build vocab
        
        # Collect all tokens from the raw data to train BPE
        all_tokens = []
        for arg in args:
            for ex in arg.examples:
                all_tokens.extend(getattr(ex, self.name))

        # Create a dummy file-like object for learn_bpe
        temp_input = io.StringIO(" ".join(all_tokens))
        temp_output = io.StringIO()
        
        # Learn BPE codes
        learn_bpe.learn_bpe(temp_input, temp_output, self.vocab_size, is_dict=False)
        self.bpe_codes = temp_output.getvalue()
        
        # Initialize BPE applier
        self.bpe = apply_bpe.BPE(io.StringIO(self.bpe_codes))

        # Apply BPE to the tokens and rebuild vocabulary
        bpe_tokens = [self.bpe.process_line(" ".join(t)).split() for t in [getattr(ex, self.name) for arg in args for ex in arg.examples]]
        
        counter = Counter()
        for tokens in bpe_tokens:
            counter.update(tokens)

        # Add special tokens
        if '<unk>' not in counter: counter['<unk>'] = 0
        if '<pad>' not in counter: counter['<pad>'] = 0
        if '<bos>' not in counter: counter['<bos>'] = 0
        if '<eos>' not in counter: counter['<eos>'] = 0

        self.vocab = Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        self.vocab.stoi['<unk>'] = 0 # Ensure UNK is 0
        self.vocab.stoi['<pad>'] = 1 # Ensure PAD is 1
        self.vocab.stoi['<bos>'] = 2 # Ensure BOS is 2
        self.vocab.stoi['<eos>'] = 3 # Ensure EOS is 3
        self.vocab.itos = [token for token, _ in sorted(self.vocab.stoi.items(), key=lambda item: item[1])]


    def preprocess(self, x):
        # Apply BPE during preprocessing
        if self.bpe:
            return self.bpe.process_line(" ".join(x)).split()
        return x # Fallback if BPE not initialized

def load_and_preprocess_data(config):
    # Initialize tokenizers
    # Using spacy for basic tokenization before BPE
    try:
        spacy_en = spacy.load('en_core_web_sm')
        spacy_de = spacy.load('de_core_news_sm')
    except OSError:
        print("Downloading spaCy models...")
        os.system("python -m spacy download en_core_web_sm")
        os.system("python -m spacy download de_core_news_sm")
        spacy_en = spacy.load('en_core_web_sm')
        spacy_de = spacy.load('de_core_news_sm')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    SRC = BPEField(codes_path=config.bpe_codes_path, vocab_size=config.bpe_vocab_size,
                   tokenize=tokenize_en, init_token='<bos>', eos_token='<eos>',
                   lower=True, include_lengths=True,
                   pad_token='<pad>', unk_token='<unk>')
    
    TRG = BPEField(codes_path=config.bpe_codes_path, vocab_size=config.bpe_vocab_size,
                   tokenize=tokenize_de, init_token='<bos>', eos_token='<eos>',
                   lower=True, include_lengths=True,
                   pad_token='<pad>', unk_token='<unk>')

    # Download WMT 2014 En-De dataset (simplified for demonstration)
    # In a real scenario, you'd download from http://www.statmt.org/wmt14/translation-task.html
    # and process the raw files.
    # For this script, we'll use Multi30k as a proxy, which is smaller but demonstrates the pipeline.
    # If you have WMT14 files, you can replace 'Multi30k' with a custom Dataset.
    from torchtext.datasets import Multi30k
    print("Loading Multi30k dataset (proxy for WMT14)...")
    train_data, valid_data, test_data = Multi30k.splits(exts=('.' + config.src_lang, '.' + config.trg_lang),
                                                        fields=(SRC, TRG),
                                                        root=config.data_dir)

    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    # Build vocabularies (BPE will be learned during this step)
    # For En-De, the paper used a shared vocabulary.
    # For En-Fr, they used separate. We'll use shared for En-De as per config.
    print("Building vocabulary (this will learn BPE codes)...")
    SRC.build_vocab(train_data.src, min_freq=2)
    TRG.build_vocab(train_data.trg, min_freq=2)

    print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

    # Create iterators
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=config.device)

    return SRC, TRG, train_iterator, valid_iterator, test_iterator

# --- 2. Transformer Model Implementation ---

# Helper function for attention masking
def get_attn_pad_mask(seq_q, seq_k):
    ''' For masking out padded region in attention
        seq_q: (batch_size, len_q)
        seq_k: (batch_size, len_k)
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(Config.TRG.vocab.stoi['<pad>']).unsqueeze(1) # (batch_size, 1, len_k)
    return pad_attn_mask.expand(batch_size, len_q, len_k) # (batch_size, len_q, len_k)

def get_subsequent_mask(seq):
    ''' For masking out subsequent info from current timestep
        seq: (batch_size, len_seq)
    '''
    attn_shape = (seq.size(1), seq.size(1))
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask.unsqueeze(0).to(Config.device) # (1, len_seq, len_seq)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout_rate=0.0):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, n_heads, seq_len, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Use 0 for mask, not True/False directly

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, dropout_rate=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(d_k, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        residual = q
        
        # 1) Do all the linear projections in batch from d_model => n_heads x d_k or d_v
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads * d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply attention on all the projected vectors in batch.
        # (batch_size, n_heads, seq_len_q, d_k) x (batch_size, n_heads, d_k, seq_len_k)
        # -> (batch_size, n_heads, seq_len_q, seq_len_k)
        output, attn = self.attention(q, k, v, mask)

        # 3) "Concat" using a view and apply a final linear.
        # (batch_size, n_heads, seq_len_q, d_v) -> (batch_size, seq_len_q, n_heads * d_v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.dropout(self.fc(output)) # Final linear layer

        # Add & Norm
        output = self.layer_norm(residual + output)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.0):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads, dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

    def forward(self, enc_input, enc_self_attn_mask):
        enc_output, enc_self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=enc_self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads, dropout_rate)
        self.enc_attn = MultiHeadAttention(d_model, n_heads, d_model // n_heads, d_model // n_heads, dropout_rate)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

    def forward(self, dec_input, enc_output, dec_self_attn_mask, dec_enc_attn_mask):
        dec_output, dec_self_attn = self.self_attn(dec_input, dec_input, dec_input, mask=dec_self_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout_rate, max_seq_len):
        super().__init__()
        self.src_word_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=Config.SRC.vocab.stoi['<pad>'])
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask):
        # src_seq: (batch_size, src_len)
        # src_mask: (batch_size, 1, src_len) or (batch_size, src_len, src_len)
        
        # Word embedding + positional embedding
        enc_