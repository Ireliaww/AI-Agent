This script implements the Transformer model and a training loop for neural machine translation, inspired by the "Attention Is All You Need" paper. It focuses on the WMT 2014 English-to-German (or English-to-French) task, using PyTorch.

**Key Features:**
1.  **Transformer Architecture:** Implements the full Transformer model, including multi-head attention, positional encoding, feed-forward networks, and encoder-decoder stacks.
2.  **Data Loading:** Utilizes `torchtext.legacy` to load WMT 2014 datasets (with a fallback to Multi30k for easier demonstration if WMT14 download fails). It includes Spacy tokenization. For true WMT 2014 reproduction, Byte-Pair Encoding (BPE) is recommended.
3.  **Hyperparameters:** Uses hyperparameters corresponding to the "base model" from the original Transformer paper, including the Noam learning rate schedule and label smoothing.
4.  **Token-based Batching:** Implements the token-based batching strategy as described in the paper, where batch size is determined by the total number of tokens rather than sentences.
5.  **Training Loop:** A standard PyTorch training loop with forward pass, loss calculation (using Label Smoothing), backward pass, and optimizer steps.
6.  **Validation Loop:** Evaluates model performance on a validation set, calculating loss and BLEU score.
7.  **Logging & Checkpointing:** Logs training progress (loss, learning rate, time) and saves model checkpoints