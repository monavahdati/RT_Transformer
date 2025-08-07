# RTTransformer Classifier

A compact Transformer-based classifier implemented in PyTorch, suitable for tabular data classification tasks.

## 📌 Overview

This repository contains the implementation of `RTTransformer`, a lightweight attention-based model that applies a transformer encoder to tabular data. It supports multiclass classification using cross-entropy loss.

## 🧠 Model Structure

- Linear layer to project inputs to `d_model`
- Stacked TransformerEncoder layers with multi-head attention
- Global average pooling across the sequence dimension
- Final classification layer for output logits

```python
RTTransformer(
    input_dim=...,
    d_model=64,
    num_heads=8,
    num_layers=4,
    num_classes=2
)
