# Dataset

## Overview

This study uses the arXiv scientific article dataset released by Cohan et al. [1]. The original dataset consists of long and structured scientific articles obtained from the arXiv repository, including article identifiers, abstracts, full-text content, and section-level information.

The dataset used in this study was derived from the original arXiv collection through article selection and additional preprocessing to support the CLiRSum multi-task scientific document generation experiments. Article titles and Introduction sections were extracted separately using the corresponding article identifiers and section-level information.

## Original Dataset Sources

The original dataset and its associated resources are available through the following sources:

- **Original dataset repository:** https://github.com/armancohan/long-summarization
- **arXiv dataset download mirror:** https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip
- **Original research paper:** https://aclanthology.org/N18-2097/
- **arXiv version of the paper:** https://arxiv.org/abs/1804.05685
- **arXiv repository:** https://arxiv.org/

## Dataset Splits

The experiments use the following data partitions:

- **Training and Validation:** 20,000 randomly selected articles, including 18,000 training articles and 2,000 validation articles.
- **Test:** 6,440 articles from the original arXiv test partition, kept separate from the training and validation data.

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
| `section_names` | List of section names. |
| `sections` | Textual contents corresponding to the article sections. |

## Dataset Preprocessing

The Introduction section was identified using the section-level information in the original dataset. The corresponding text was extracted and removed from the complete article text to construct `article_text_no_intro`. Article titles were retrieved separately from the arXiv API using the corresponding article identifiers.

## Dataset Variants

Two dataset variants were constructed to investigate different source-input selection strategies under BART's input-length constraints:

- **Dataset 1:** Uses beginning-of-text truncation.
- **Dataset 2:** Selects input text from four article segments for the article-based tasks (T1 and T2).

## Citation

If you use the original arXiv dataset, please cite the following publication:

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018). A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents. *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 615–621.

https://doi.org/10.18653/v1/N18-2097
