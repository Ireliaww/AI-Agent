This research report outlines the implementation and training script for a Transformer model, based on the findings described in the provided analysis of the "Attention Is All You Need" paper. The script aims to reproduce the core experimental setup and methodology for the WMT 2014 English-to-German and English-to-French translation tasks.

**Key Features of the Training Script:**

1.  **Transformer Model Implementation:** A PyTorch implementation of the Transformer architecture, including Multi-Head Attention, Position-wise Feed-Forward Networks, Positional Encoding, Encoder, and Decoder layers.
2.  **Data Loading:** Uses `torchtext` for simplified data loading, tokenization (using `spacy`), vocabulary building, and batching. *Note: For full WMT 2014 reproducibility, Byte-Pair Encoding (BPE) and specific preprocessing steps (e.g., `multi-bleu.perl` for evaluation) would be required, which are beyond the scope of this self-contained script and are acknowledged as simplifications.*
3.  **Hyperparameters:** Incorporates the key hyperparameters identified in the paper for the Transformer model, such as `d_model`, `n_layers`, `n_heads`, `dropout`, and `label_smoothing`.
4.  **Optimizer and Learning Rate Schedule:** Implements the custom Adam optimizer with the "Noam" learning rate schedule (warmup followed by inverse square root decay).
5.  **Loss Function:** Uses `nn.KLDivLoss` with label smoothing as specified in the paper.