The following is a complete PyTorch training script for the Transformer model, designed to reproduce the key findings of the "Attention Is All You Need" paper. It includes data loading with SentencePiece Byte-Pair Encoding (BPE), the full Transformer architecture, a custom learning rate scheduler, label smoothing, and a training/validation loop with logging and checkpointing.

**Key Features:**
1.  **Data Loading:** Utilizes `torchtext.datasets.WMT14` to fetch raw data and trains/applies `sentencepiece` for BPE tokenization. Implements custom token-based batching.
2.  **Transformer Architecture:** Full implementation of Positional Encoding, Multi-Head Attention, Position-wise Feed-Forward Networks, Residual Connections, Layer Normalization, Encoder, and Decoder.
3.  **Hyperparameters:** Configurable for both "Base" and "Big" Transformer models as described in the paper.
4.  **Learning Rate Scheduler:** Implements the `NoamOpt` custom learning rate schedule with warm-up.
5.  **Loss Function:** Uses `CrossEntropyLoss` with label smoothing.
6.  **Training Loop:** Includes standard forward/backward passes, optimizer steps, and epoch-based iteration.
7.  **Validation:** Evaluates the model on a validation set and calculates BLEU scores using `sacrebleu`.
8.  **Logging & Checkpointing:** Saves the best model based on validation BLEU score and logs training progress.

---

```python
import torch
import torch.nn as nn