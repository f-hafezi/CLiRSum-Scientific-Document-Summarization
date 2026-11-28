# CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization

This repository presents **CLiRSum**, a BART-Large-based multi-task framework for scientific document summarization.

## Tasks

The framework jointly addresses four tasks:

1. **T1:** Article-to-Introduction
2. **T2:** Article-to-Abstract
3. **T3:** Introduction-to-Abstract
4. **T4:** Abstract-to-Title

## Model Architecture

- **Backbone:** BART-Large
- **Framework:** Hugging Face Transformers
- **Training Strategy:** Simultaneous Learning (Sim)
- **Architecture:** Shared Encoder-Decoder
- **Loss:** Task-Weighted Cross-Entropy

### Task-Specific Prefixes

```text
generate_introduction:
generate_abstract:
generate_abstract_from_intro:
generate_title:
```

## Training Strategy

The framework incorporates:

- **T1-Based Initialization:** Initial training on Article-to-Introduction.
- **Joint Multi-Task Learning:** Training across all four tasks.
- **Task-Aware Loss Weighting:** Adaptive balancing of task contributions.
- **Input Selection Strategies:** Evaluation of different dataset variants, including preserving the Introduction in the T2 input.

## Trained Models

Two model configurations are provided:

| Directory | Description |
|---|---|
| `T1-Initialized/` | Model initialized through single-task training on T1. |
| `Model_T1-Init + T2-AddIntro/` | Model incorporating T1 initialization and the T2 input configuration with the Introduction preserved. |


## Dataset

The dataset used in this study is publicly available through Zenodo and GitHub:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24592101.svg)](https://doi.org/10.5281/zenodo.24592101)
- **Zenodo (DOI):** https://doi.org/10.5281/zenodo.24592101
- **GitHub:** [https://github.com/f-hafezi/CLiRSum-Dataset](https://github.com/f-hafezi/CLiRSum-Scientific-Document-Summarization/blob/main/Dataset/README.md)

For details on the dataset structure, preprocessing, and experimental variants, please refer to the dataset repository.

## Citation

If you use this repository, its code, or the provided models in your research, please cite our work:   

## License

Please refer to the repository license for terms of use.
