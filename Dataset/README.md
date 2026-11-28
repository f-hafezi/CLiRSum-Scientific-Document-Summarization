# CLiRSum-Dataset
# CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization

**Authors:** Fahimeh Hafezi¹, Marziea Rahimi¹*, and Zahra Rahimi²

**Corresponding Author:** Marziea Rahimi  
**Email:** Marziea.rahimi@shahroodut.ac.ir

**Co-authors:**  
- Fahimeh Hafezi — fhafezi@shahroodut.ac.ir
- Zahra Rahimi — za.rah@atu.ac.ir

---

## Overview

This repository provides the processed dataset and associated resources for the study entitled *"CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization."*

The original data were obtained from the arXiv scientific article dataset released by Cohan et al. [1]. This collection contains scientific articles from multiple research fields, including article identifiers, abstracts, full-text content, and section-level information.

Based on the original collection, a processed dataset was prepared for the CLiRSum multi-task scientific document summarization experiments. Additional preprocessing was performed to extract article titles and Introduction sections and construct the textual fields required for the experimental tasks.

## Original Dataset Sources

The original dataset and its associated research publication are available through the following sources:

- **Original dataset download:** [arXiv Dataset](https://archive.org/download/armancohan-long-summarization-paper-code/arxiv-dataset.zip)
- **Original research paper:** [Cohan et al. (2018)](https://aclanthology.org/N18-2097/)
- **Original GitHub repository:** [long-summarization](https://github.com/armancohan/long-summarization)

## Processed Dataset Used in This Study

The processed dataset generated from the original arXiv collection and used in this study is available at:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24592101.svg)](https://doi.org/10.5281/zenodo.24592101)

**Dataset DOI:** https://doi.org/10.5281/zenodo.24592101


The training and validation subset contains 20,000 randomly selected articles, comprising 18,000 training articles and 2,000 validation articles.

The original arXiv test partition, containing 6,440 articles, was kept separate and used for testing.

## Dataset Structure

The processed dataset is organized in CSV format and contains the following columns:

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

The extracted Introduction was then searched for within the complete article text and removed to construct `article_text_no_intro`. Article titles were retrieved separately from the arXiv API using their corresponding article identifiers.

## Dataset Variants

Two dataset variants were constructed to investigate different source-input selection strategies under BART's input-length constraints:

- **Dataset 1:** Uses beginning-of-text truncation to construct model inputs.
- **Dataset 2:** Selects input text from four article segments for the article-based tasks (T1 and T2).

These variants were used to examine the effects of different input-selection strategies on the summarization tasks.

## Citation

If you use the original arXiv dataset, please cite the following publication:

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018). A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents. *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 615–621.

https://doi.org/10.18653/v1/N18-2097

If you use the processed dataset or associated resources provided in this repository, please also cite the CLiRSum study.

Fahimeh Hafezi, Marziea Rahimi, & Zahra Rahimi. (2026). Dataset: CLiRSum: Exploiting Cross-Level Information Relationships for Multi-Task Scientific Document Summarization [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.24592101

## Contact

For questions regarding the dataset or associated resources, please contact:

- **Fahimeh Hafezi:** fhafezi@shahroodut.ac.ir
- **Marziea Rahimi:** Marziea.rahimi@shahroodut.ac.ir
- **Zahra Rahimi:** za.rah@atu.ac.ir
