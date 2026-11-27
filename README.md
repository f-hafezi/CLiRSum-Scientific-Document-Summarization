# CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization

This repository presents **CLiRSum**, a multi-task learning framework based on **BART (Bidirectional and Auto-Regressive Transformers)** for scientific document summarization and structured text generation.

The framework exploits cross-level information relationships within scientific documents through a shared encoder-decoder architecture, enabling multiple related generation tasks to be learned within a unified model.

## Tasks

CLiRSum is designed to perform four complementary NLP tasks:

- **T1 – Introduction Generation:** Generate an introduction from a full scientific article.
- **T2 – Abstract Generation:** Generate an abstract from a full scientific article.
- **T3 – Abstract Generation from Introduction:** Generate an abstract using the introduction as input.
- **T4 – Title Generation:** Generate a title from an abstract.

## Methodology

The framework is built upon a shared BART architecture and incorporates:

- Task-specific prefixes for identifying generation objectives.
- Task-aware loss weighting for balancing multiple learning objectives.
- Multi-stage training, combining task-specific initialization with joint multi-task learning.
- Task-dependent sequence lengths and generation configurations.

The training strategy first initializes the model using the introduction generation task (T1), followed by multi-task training involving the four generation objectives.

## Model Architecture

- **Backbone:** BART
- **Framework:** Hugging Face Transformers
- **Learning Paradigm:** Multi-Task Sequence-to-Sequence Learning
- **Training Objective:** Task-Weighted Cross-Entropy Loss
- **Architecture:** Shared Encoder-Decoder

Task-specific prefixes:

```text
generate_introduction:
generate_abstract:
generate_abstract_from_intro:
generate_title:
```

## Trained Models

Two trained model configurations are provided in this repository:

| Directory | Description |
|---|---|
| `T1-Initialized/` | Model initialized through single-task training on T1 (Introduction Generation). |
| `Model_T1-Init + T2-AddIntro/` | Final model configuration incorporating T1 initialization and the additional T2-related training setup. |

Each directory contains the corresponding model artifacts and associated files required for using the trained model.



## Research Focus

CLiRSum investigates how cross-level information relationships and shared representations can support scientific document summarization. By jointly learning article-to-section and section-to-section generation objectives, the framework explores a unified approach to generating interconnected components of scientific documents.

## License

Please refer to the repository license for terms of use.
