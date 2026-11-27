# CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization

**Authors:** Fahimeh Hafezi¹, Marziea Rahimi¹*, and Zahra Rahimi²

**Co-author Contact:** Fahimeh Hafezi  
**Email:** fhafezi@shahroodut.ac.ir

**Corresponding Author:** Marziea Rahimi  
**Email:** Marziea.rahimi@shahroodut.ac.ir

**Co-author Contact:** Zahra Rahimi  
**Email:** za.rah@atu.ac.ir

---

# Dataset

## Overview

This repository provides the processed dataset used in the study entitled *"CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization."*

The original data were obtained from the arXiv scientific article dataset released by Cohan et al. [1]. This collection consists of scientific articles from multiple research fields and includes article identifiers, abstracts, full-text content, and section-level information.

Based on the original collection, a processed dataset was prepared for the CLiRSum multi-task scientific document summarization experiments. Additional preprocessing was performed to extract article titles and Introduction sections and construct the textual fields required for the experimental tasks.

## Dataset Sources

The original dataset and its associated resources are available through the following sources:

- **Original dataset repository:** https://github.com/armancohan/long-summarization
- **Original arXiv dataset download mirror:** https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip
- **Original research paper:** https://aclanthology.org/N18-2097/
- **arXiv version of the paper:** https://arxiv.org/abs/1804.05685
- **arXiv repository:** https://arxiv.org/

## Processed Dataset Used in This Study

The processed dataset generated from the original arXiv collection and used in this study is available at:

**[Download the Processed CLiRSum Dataset](YOUR_PUBLIC_DATASET_LINK)**

The training and validation subset contains 20,000 randomly selected articles, comprising 18,000 training articles and 2,000 validation articles.

The original arXiv test partition, containing 6,440 articles, was kept separate and used for testing.

## Dataset Structure

The processed dataset is provided in CSV format and contains the following columns:

| Column | Description |
|---|---|
| `article_id` | Unique identifier of the scientific article. |
| `abstract_text` | Abstract of the article. |
| `title` | Article title retrieved using the arXiv API. |
| `introduction_text` | Introduction section extracted from the section-level information. |
| `article_text_no_intro` | Article text with the Introduction section removed. |
| `article_text` | Complete article text. |
| `section_names` | List of section names identified in the article. |
| `sections` | Textual contents corresponding to the article sections. |

## Dataset Preprocessing

The Introduction section was identified using the section-level information in the original dataset. Specifically, the section-name list was searched to locate the Introduction, and the corresponding text was extracted from the section-text list.

The extracted Introduction was then searched within the complete article text and removed to construct `article_text_no_intro`. Article titles were retrieved separately from the arXiv API using their corresponding article identifiers.

## Dataset Variants

Two dataset variants were constructed to investigate different source-input selection strategies under BART's input-length constraints:

- **Dataset 1:** Uses beginning-of-text truncation to construct model inputs.
- **Dataset 2:** Selects input text from four article segments for the article-based tasks (T1 and T2).

These variants were used to investigate the effects of input selection strategies on input-length statistics and semantic coverage.

## Citation

If you use the original arXiv dataset, please cite the following publication:

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018). A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents. *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 615–621.

https://doi.org/10.18653/v1/N18-2097

If you use the processed dataset or code provided in this repository, please also cite the CLiRSum study.

## Contact

For questions regarding the dataset or the associated code, please contact:

- **Fahimeh Hafezi:** fhafezi@shahroodut.ac.ir
- **Marziea Rahimi:** Marziea.rahimi@shahroodut.ac.ir
- **Zahra Rahimi:** za.rah@atu.ac.ir
