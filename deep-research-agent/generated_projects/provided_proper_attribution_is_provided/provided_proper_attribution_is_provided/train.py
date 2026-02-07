This script aims to reproduce the training setup for the Transformer model, focusing on the WMT 2014 English-to-German and English-to-French translation tasks, as described in the "Attention Is All You Need" paper.

**Important Considerations and Limitations:**

1.  **Hyperparameters:** The provided analysis *does not list specific numerical hyperparameters* (e.g., `d_model`, `n_layers`, learning rate schedule parameters). Therefore, the hyperparameters used in this script are **inferred from the original "Attention Is All You Need" paper** for the "big" Transformer model. This is a critical distinction.
2.  **Data Loading (WMT 2014):** Loading the full WMT 2014 datasets with `torchtext` can be very time-consuming and memory-intensive, especially for vocabulary building and tokenization. The original paper used byte-pair encoding (BPE) for subword tokenization, which is more robust for translation tasks. For simplicity and to make the script runnable without extensive setup, this script uses `spacy` for word-level tokenization. For true reproduction, BPE tokenization (e.g., using `subword-nmt`) would be necessary.
3.  **Batching Strategy:** The original Transformer paper used a "number of token pairs per batch" strategy to minimize padding. This script uses `BucketIterator` with a fixed `batch_size`, which is a common approximation but not identical to the original paper's dynamic batching.
4.  **Full Transformer Implementation:** While PyTorch provides `nn.TransformerEncoder` and `nn.TransformerDecoder`, we need to wrap them with embedding layers, positional encodings, and a final linear projection to form the complete Transformer model as described in the paper.
5.  **BLEU Score Calculation:** `sacrebleu` is used for BLEU score calculation, which is the standard for reproducibility in machine translation research.
6.  **Computational Resources:** Training the "big" Transformer model on WMT 2014 datasets requires significant computational resources (e.g., multiple P100 GPUs for several days, as mentioned in the paper). Running this script on a single GPU or CPU will be very slow and likely won't achieve the reported state-of-the-art results.

---

### PyTorch Training Script for Transformer Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import math
import time
import os
from datetime import datetime

# Install necessary libraries if not already installed
try:
    import spacy
except ImportError:
    print("Installing spacy...")
    os.system("pip install spacy")
    os.system("python -m spacy download en_core_web_sm")
    os.system("python -m spacy download de_core_news_sm")
    os.system("python -m spacy download fr_core_news_sm")
    import spacy

try:
    import torchtext
    from torchtext.data import Field, BucketIterator
    from torchtext.datasets import WMT14
except ImportError:
    print("Installing torchtext...")
    os.system("pip install torchtext==0.6.0") # Specific version compatible with Field/BucketIterator
    from torchtext.data import Field, BucketIterator
    from torchtext.datasets import WMT14

try:
    import sacrebleu
except ImportError:
    print("Installing sacrebleu...")
    os.system("pip install sacrebleu")
    import sacrebleu


# --- 1. Configuration and Hyperparameters (Inferred from "Attention Is All You Need" paper for Big Transformer) ---
class Hparams:
    # Model parameters for the "Big" Transformer
    d_model = 1024          # Dimension of embeddings and Transformer layers (512 for base, 1024 for big)
    n_heads = 16            # Number of attention heads (8 for base, 16 for big)
    n_layers = 6            # Number of encoder and decoder layers
    d_ff = 4096             # Dimension of the feed-forward network (2048 for base, 4096 for big)
    dropout = 0.1           # Dropout probability

    # Training parameters
    src_lang = 'en'
    tgt_lang = 'de' # Can be 'fr' for English-to-French
    epochs = 10             # For demonstration; paper trained for much longer
    batch_size = 128        # Number of samples per batch (paper used ~25000 token pairs)
    clip = 1.0              # Gradient clipping
    label_smoothing = 0.1   # Label smoothing epsilon
    max_seq_len = 100       # Maximum sequence length to consider (truncate longer sequences)

    # Learning rate schedule parameters (Adam optimizer)
    lr_warmup_steps = 4000  # Number of warmup steps
    lr_multiplier = 1.0     # Multiplier for the base learning rate formula

    # Logging and Checkpointing
    log_interval = 100      # Log every N batches
    checkpoint_dir = "checkpoints"
    tensorboard_log_dir = "runs"
    model_name = f"transformer_big_{tgt_lang}"

hparams = Hparams()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Ensure checkpoint directory exists
os.makedirs(hparams.checkpoint_dir, exist_ok=True)
os.makedirs(hparams.tensorboard_log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(os.path.join(hparams.tensorboard_log_dir, datetime.now().strftime('%Y%m%d-%H%M%S')))

# --- 2. Data Loading and Preprocessing ---
print("Loading Spacy tokenizers...")
try:
    spacy_en = spacy.load('en_core_web_sm')
    spacy_src = spacy.load(f'{hparams.src_lang}_core_web_sm') # Assuming en_core_web_sm for source
    spacy_tgt = spacy.load(f'{hparams.tgt_lang}_core_news_sm') # de_core_news_sm or fr_core_news_sm for target
except OSError:
    print(f"Spacy model for {hparams.tgt_lang} not found. Attempting to download...")
    os.system(f"python -m spacy download {hparams.tgt_lang}_core_news_sm")
    spacy_tgt = spacy.load(f'{hparams.tgt_lang}_core_news_sm')
    print("Spacy model downloaded and loaded.")


def tokenize_src(text):
    return [tok.text for tok in spacy_src.tokenizer(text)][:hparams.max_seq_len]

def tokenize_tgt(text):
    return [tok.text for tok in spacy_tgt.tokenizer(text)][:hparams.max_seq_len]

# Define fields for source and target languages
SRC = Field(tokenize=tokenize_src,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

TRG = Field(tokenize=tokenize_tgt,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

print(f"Loading WMT 2014 {hparams.src_lang}-to-{hparams.tgt_lang} dataset...")
# Load WMT 2014 dataset
# Note: This can take a very long time and consume a lot of memory.
# For quick testing, consider using torchtext.datasets.Multi30k.
train_data, valid_data, test_data = WMT14.splits(exts=(f'.{hparams.src_lang}', f'.{hparams.tgt_lang}'),
                                                 fields=(SRC, TRG),
                                                 root='.data',
                                                 train=f'wmt14_de_en/wmt14_de_en_train.tok.shuffled', # Adjust path if needed
                                                 validation=f'wmt14_de_en/newstest2013.tok',
                                                 test=f'wmt14_de_en/newstest2014.tok',
                                                 # For French, replace 'de_en' with 'fr_en' and adjust test set
                                                 # train=f'wmt14_fr_en/wmt14_fr_en_train.tok.shuffled',
                                                 # validation=f'wmt14_fr_en/newstest2013.tok',
                                                 # test=f'wmt14_fr_en/newstest2014.tok'
                                                 )
print(f"Dataset loaded. Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of test examples: {len(test_data.examples)}")

print("Building vocabulary...")
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"Source vocabulary size: {len(SRC.vocab)}")
print(f"Target vocabulary size: {len(TRG.vocab)}")

# Create iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=hparams.batch_size,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src)
)
print("Data iterators created.")

# --- 3. Model Definition (Transformer Architecture) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len, pad_idx):
        super().__init__()

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=n_heads,
                                          num_encoder_layers=n_layers,
                                          num_decoder_layers=n_layers,
                                          dim_feedforward=d_ff,
                                          dropout=dropout,
                                          batch_first=True)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt = [batch size, tgt len]
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # tgt_pad_mask = [batch size, 1, 1, tgt len]

        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()
        # tgt_sub_mask = [tgt len, tgt len]

        tgt_mask = tgt_pad_mask & ~tgt_sub_mask
        # tgt_mask = [batch size, 1, tgt len, tgt len]
        return tgt_mask

    def forward(self, src, tgt):
        # src = [batch size, src len]
        # tgt = [batch size, tgt len]

        src_mask = self.transformer.generate_square_subsequent_mask(src.shape[1]).to(device)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)

        src_padding_mask = (src == self.pad_idx).to(device)
        tgt_padding_mask = (tgt == self.pad_idx).to(device)

        # PyTorch's Transformer expects (seq_len, batch_size, d_model) for input, and masks are for (seq_len, seq_len)
        # But here we are using batch_first=True, so input is (batch_size, seq_len, d_model)
        # The masks for batch_first=True are expected to be (batch_size, seq_len) for padding mask
        # and (seq_len, seq_len) for attention mask.

        src_embedded = self.dropout(self.pos_encoder(self.src_tok_emb(src) * math.sqrt(hparams.d_model)))
        tgt_embedded = self.dropout(self.pos_encoder(self.tgt_tok_emb(tgt) * math.sqrt(hparams.d_model)))

        # Adjust masks for PyTorch nn.Transformer with batch_first=True
        # src_mask and tgt_mask are causal masks (square subsequent) for attention
        # src_padding_mask and tgt_padding_mask are boolean masks for padding

        output = self.transformer(src_embedded, tgt_embedded,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=src_padding_mask) # Encoder output memory padding mask

        output = self.fc_out(output)
        # output = [batch size, tgt len, tgt vocab size]
        return output

# --- 4. Optimizer and Loss Function ---
# Custom Learning Rate Scheduler (from "Attention Is All You Need" paper)
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

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.model_size**(-0.5) * \
               min(step**(-0.5), step * self.warmup**(-1.5)) * \
               hparams.lr_multiplier

# Label Smoothing Loss (from "Attention Is All You Need" paper)
class LabelSmoothingLoss(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) # -2 for <sos> and <eos>
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

# Initialize model
SRC_VOCAB_SIZE = len(SRC.vocab)
TRG_VOCAB_SIZE = len(TRG.vocab)
PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Transformer(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, hparams.d_model,
                    hparams.n_heads, hparams.n_layers, hparams.d_ff,
                    hparams.dropout, hparams.max_seq_len, PAD_IDX).to(device)

# Initialize weights
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

# Optimizer and Loss
optimizer = NoamOpt(hparams.d_model, hparams.lr_warmup_steps,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = LabelSmoothingLoss(TRG_VOCAB_SIZE, PAD_IDX, hparams.label_smoothing).to(device)

print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

# --- 5. Training and Evaluation Functions ---
def train(model, iterator, optimizer, criterion, clip, writer, step_count):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # trg input is <sos>, x1, x2, ..., xn
        # trg output is x1, x2, ..., xn, <eos>
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]

        optimizer.optimizer.zero_grad() # Use optimizer.optimizer for NoamOpt
        output = model(src, trg_input)

        # output = [batch size, trg len - 1, output dim]
        # trg_output = [batch size, trg len - 1]

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_output = trg_output.contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg_output = [batch size * trg len - 1]

        loss = criterion(output, trg_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        step_count += 1

        epoch_loss += loss.item()

        if i % hparams.log_interval == 0 and i > 0:
            avg_batch_loss = epoch_loss / (i + 1)
            writer.add_scalar('Training Loss/Batch', avg_batch_loss, step_count)
            writer.add_scalar('Learning Rate', optimizer.rate(), step_count)
            print(f"  Batch: {i:05d} | Step: {step_count:07d} | Train Loss: {avg_batch_loss:.3f} | LR: {optimizer.rate():.6f}")

    return epoch_loss / len(iterator), step_count

def evaluate(model, iterator, criterion, TRG_vocab, writer, step_count, max_examples=1000):
    model.eval()
    epoch_loss = 0
    translations = []
    references = []
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if i * hparams.batch_size > max_examples: # Limit evaluation for faster feedback
                break

            src = batch.src.to(device)
            trg = batch.trg.to(device)

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            output = model(src, trg_input)
            
            output_dim = output.shape[-1]
            output_flat = output.contiguous().view(-1, output_dim)
            trg_output_flat = trg_output.contiguous().view(-1)

            loss = criterion(output_flat, trg_output_flat)
            epoch_loss += loss.item()

            # Generate translations for BLEU calculation
            # Greedy decoding for simplicity, beam search is used in paper
            start_token = TRG_vocab.stoi[TRG.init_token]
            end_token = TRG_vocab.stoi[TRG.eos_token]
            
            # Initialize target sequence with <sos> token
            translated_tokens = torch.full((src.shape[0], 1), start_token, dtype=torch.long, device=device)

            for t in range(hparams.max_seq_len - 1): # Max length for decoding
                # Get model output for current translated sequence
                prediction = model(src, translated_tokens) # [batch_size, current_len, vocab_size]
                
                # Get the last token's prediction and take the argmax
                next_token_logits = prediction[:, -1, :] # [batch_size, vocab_size]
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(1) # [batch_size, 1]
                
                # Append to translated sequence
                translated_tokens = torch.cat((translated_tokens, next_token), dim=1)
                
                # Stop if all sequences have generated <eos>
                if ((next_token == end_token) | (translated_tokens == end_token)).all():
                    break
            
            # Convert token IDs to words for BLEU
            for j in range(src.shape[0]):
                pred_tokens = [TRG_vocab.itos[tok_idx] for tok_idx in translated_tokens[j] if tok_idx not in [start_token, end_token, PAD_IDX]]
                ref_tokens = [TRG_vocab.itos[tok_idx] for tok_idx in trg[j] if tok_idx not in [start_token, end_token, PAD_IDX]]

                translations.append(' '.join(pred_tokens))
                references.append([' '.join(ref_tokens)]) # sacrebleu expects list of lists for references

    # Calculate BLEU score
    if len(translations) > 0:
        bleu = sacrebleu.corpus_bleu(translations, references)
        print(f"  BLEU Score: {bleu.score:.2f}")
        writer.add_scalar('Validation/BLEU', bleu.score, step_count)
    else:
        bleu = None
        print("  No translations generated for BLEU calculation.")

    return epoch_loss / len(iterator), bleu.score if bleu else 0.0

# --- 6. Main Training Loop ---
best_valid_loss = float('inf')
best_bleu_score = 0.0
global_step = 0

print("\nStarting training...")
for epoch in range(hparams.epochs):
    start_time = time.time()

    train_loss, global_step = train(model, train_iterator, optimizer, criterion, hparams.clip, writer, global_step)
    valid_loss, valid_bleu = evaluate(model, valid_iterator, criterion, TRG.vocab, writer, global_step, max_examples=2000)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    writer.add_scalar('Training Loss/Epoch', train_loss, global_step)
    writer.add_scalar('Validation Loss/Epoch', valid_loss, global_step)

    print(f"Epoch: {epoch+1:02d} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f}")
    print(f"\t Val. BLEU: {valid_bleu:.2f}")

    # Checkpointing based on validation BLEU score
    if valid_bleu > best_bleu_score:
        best_bleu_score = valid_bleu