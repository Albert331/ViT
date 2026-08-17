# Vision Transformer (ViT) From Scratch

A Vision Transformer built from scratch in PyTorch for image classification, implementing the core Transformer architecture without using a pretrained ViT or high-level vision model.

## Overview

This model takes an RGB image, divides it into fixed-size patches, projects each patch into an embedding space, and processes the resulting sequence using Transformer encoder blocks.

The implementation was built from first principles to understand how Vision Transformers operate internally, including patch embeddings, learnable class tokens, positional embeddings, multi-head self-attention, residual connections, layer normalization, and feed-forward networks.

## Architecture

The model follows the original Vision Transformer architecture, treating image patches as a sequence of tokens.

**Patch Embedding**

* Input image is divided into non-overlapping patches
* Each patch is flattened and projected into a fixed-dimensional embedding
* Patch embeddings form the input token sequence for the Transformer

**Token and Positional Embeddings**

* A learnable `[CLS]` token is prepended to the patch sequence
* Learnable positional embeddings are added to retain spatial information
* The resulting sequence is passed through the Transformer encoder

**Transformer Encoder**

* 6 Transformer encoder blocks
* Embedding dimension: 128
* Each block consists of:

  * LayerNorm
  * Multi-Head Self-Attention
  * Residual connection
  * LayerNorm
  * Feed-Forward MLP
  * Residual connection

**Classification Head**

* The final `[CLS]` token representation is extracted
* Layer normalization is applied
* A linear layer maps the representation to the target classes

**Output**

```text
[batch, num_classes]
```

## Self-Attention

The core operation of the Transformer is scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

Multi-head attention allows the model to learn relationships between different image patches simultaneously.

Unlike convolutional layers, attention provides direct interactions between patches regardless of their spatial distance, allowing the model to build a global representation of the image.

## Transformer Block

Each encoder block follows the standard pre-normalization Transformer structure:

```text
Input
  |
  v
LayerNorm
  |
  v
Multi-Head Self-Attention
  |
  v
Residual Connection
  |
  v
LayerNorm
  |
  v
Feed-Forward Network
  |
  v
Residual Connection
```

The feed-forward network operates independently on each token after the attention operation.

## Training Pipeline

```text
RGB Image
    |
    v
Image Preprocessing
    |
    v
Patch Embedding
    |
    v
[CLS] Token + Positional Embedding
    |
    v
6 Transformer Encoder Blocks
    |
    v
[CLS] Representation
    |
    v
LayerNorm
    |
    v
Linear Classification Head
    |
    v
Class Prediction
```

## Implementation Details

| Component             | Configuration             |
| --------------------- | ------------------------- |
| Framework             | PyTorch                   |
| Input                 | RGB images                |
| Embedding dimension   | 128                       |
| Transformer blocks    | 6                         |
| Positional embeddings | Learnable                 |
| Class token           | Learnable                 |
| Attention             | Multi-Head Self-Attention |
| Normalization         | LayerNorm                 |
| Activation            | ReLU/GELU                 |
| Classification        | Linear layer              |

## Why Build ViT From Scratch?

The purpose of this project was not simply to use an existing Vision Transformer implementation, but to understand how the architecture is constructed internally.

The implementation focuses on understanding:

* How an image can be represented as a sequence of tokens
* How patch embeddings are generated
* Why positional embeddings are required
* How the `[CLS]` token is used for classification
* How queries, keys, and values interact inside self-attention
* How multi-head attention captures relationships between image patches
* How residual connections and normalization stabilize Transformer training
* How multiple Transformer blocks build increasingly expressive representations

## Key Design Decisions

* Patch extraction and embedding are implemented directly rather than using a pretrained ViT.
* The `[CLS]` token is learned jointly with the rest of the model parameters.
* Positional embeddings are learned rather than using fixed sinusoidal encodings.
* Transformer blocks are implemented as modular PyTorch modules and stacked using `nn.ModuleList`.
* The classification head operates only on the final `[CLS]` representation.

## Limitations

This implementation is intentionally small compared with production-scale Vision Transformers.

The embedding dimension and number of Transformer blocks are significantly smaller than the original ViT-Base configuration, making the model more practical to train locally while keeping the core architecture intact.

Because the model is trained from scratch, performance is also highly dependent on the size and quality of the training dataset, preprocessing, and optimization configuration.

## Future Improvements

* Add configurable patch size and image resolution
* Add attention visualization
* Add training and validation loss curves
* Track additional classification metrics
* Experiment with different embedding dimensions
* Experiment with different numbers of attention heads and Transformer blocks
* Add data augmentation
* Compare against a CNN baseline
* Implement different positional encoding strategies
* Experiment with larger datasets and model configurations

## References

* Dosovitskiy et al. — [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
* Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Author

**Albert B. V.**

B.Tech Computer Science — AI & ML

[GitHub](https://github.com/Albert331)
