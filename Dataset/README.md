# Dataset

## Overview

This study uses the arXiv scientific article dataset, a large-scale collection of research papers from multiple scientific fields. The dataset contains scientific articles together with their abstracts, titles, introductions, full-text content, and section-level information.

The dataset used in this study was prepared from the original arXiv dataset and processed to support the multi-task scientific document generation experiments.

## Dataset Sources and Splits

The dataset used in the experiments consists of the following subsets:

- **Training and Validation Dataset:** 20,000 randomly selected articles from the original arXiv dataset, divided into 18,000 training articles and 2,000 validation articles.
- **Test Dataset:** 6,440 articles from the original arXiv test partition, kept separate from the training and validation data.

The 20,000-article dataset used for training and validation is available through the following link:

[Download the 20,000-Article Dataset](https://drive.google.com/file/d/1SA3Y-PmdO3RdTmr6WXqEqhiqetE1E5WN/view?usp=sharing)

The test dataset is derived from the original arXiv test partition and contains the same data fields as the training and validation dataset.

## Dataset Structure

The dataset is provided in CSV format and contains the following columns:

| Column | Description |
|---|---|
| `article_id` | Unique identifier of the scientific article. |
| `abstract_text` | Abstract of the article. |
| `title` | Title of the scientific article. |
| `introduction_text` | Extracted Introduction section of the article. |
| `article_text_no_intro` | Article text with the Introduction section removed. |
| `article_text` | Complete article text. |
| `section_names` | List of section names identified in the article. |
| `sections` | List of textual contents corresponding to the article sections. |

## Dataset Preprocessing

The Introduction section was identified using the section-level information provided in the original dataset. Specifically, the section-name list was searched to locate the Introduction, and the corresponding text was extracted from the section-text list.

The extracted Introduction was subsequently searched within the complete article text, and the matching text was removed to construct `article_text_no_intro`. The extracted Introduction was retained separately as the target text for the introduction-generation task and as an additional input for the abstract-generation task in the T2-AddIntro variants.

The resulting dataset provides the textual fields required for the multi-task generation experiments, including article-to-introduction generation, article-to-abstract generation, introduction-to-abstract generation, and abstract-to-title generation.

## Dataset Variants

Two dataset variants were constructed to investigate the effects of different source-input selection strategies under the input-length constraints of BART:

- **Dataset 1:** Uses beginning-of-text truncation to construct model inputs.
- **Dataset 2:** Selects input text from four segments of the article for the article-based tasks (T1 and T2).

The two variants are used to investigate input selection strategies and their effects on input-length statistics and semantic coverage.

## Data Availability

The training and validation dataset containing 20,000 articles is available through the Google Drive link provided above.

The test dataset consists of the original arXiv test partition containing 6,440 articles and is maintained separately from the training and validation data.

## Citation

If you use this dataset or the associated code in your research, please cite the corresponding publication:

*[Add the paper citation here after publication or acceptance.]*
