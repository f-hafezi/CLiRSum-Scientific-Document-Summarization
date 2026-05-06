Multi-Task Text Generation with BART (Introduction, Abstract, Title Generation)

This repository implements a multi-task learning framework based on BART (Bidirectional and Auto-Regressive Transformers) for structured scientific text generation.

The model is trained to perform four related NLP tasks jointly:

- Generate Introduction from full article
- Generate Abstract from full article
- Generate Abstract from Introduction
- Generate Title from Abstract

Instead of training separate models for each task, we use a single shared encoder-decoder architecture (BART) with:

- Task-specific prefixes
- Task-aware loss weighting
- Length-aware decoding
- Multi-stage training (single-task → multi-task)

This improves:
- Generalization
- Parameter efficiency
- Cross-task knowledge transfer



# Model Architecture

We use:

- Backbone: BART (Facebook AI)
- Framework: HuggingFace Transformers
- Training style: Multi-task seq2seq learning
- Loss: Weighted Cross Entropy per task

Each task is conditioned using a prefix token strategy:

generate_introduction
generate_abstract
generate_abstract_from_intro
generate_title




# Training Strategy

Stage 1: Single-Task Pretraining
- Train only Task 0 (Introduction generation)
- Stabilizes encoder understanding of long scientific text

Stage 2: Multi-Task Learning
- Joint training on all tasks
- Weighted loss per task:


task_weights = {
    0: 2.0,  # Introduction (most important)
    1: 1.5,  # Abstract
    2: 1.0,  # Abstract from intro
    3: 0.5   # Title generation
}

# Evaluation

src/evaluation.py

## Appendix
See appendix.md for experimental results tables.


