
# Ablation Study

## Table A3. Ablation Study of Model Components on Tasks T1–T4

| Approach | T1 (R1/R2/RL/BERT-F1) | T2 (R1/R2/RL/BERT-F1) | T3 (R1/R2/RL/BERT-F1) | T4 (R1/R2/RL/BERT-F1) |
|---|---|---|---|---|
| T1-Init + T2-AddIntro (Original) | **0.41/0.13/0.20/0.84** | **0.43/0.17/0.25/0.86** | **0.39/0.14/0.23/0.85** | **0.46/0.26/0.41/0.88** |
| T2-AddIntro (w/o T1) | 0.37/0.13/0.18/0.83 | **0.43/0.17/0.24/0.84** | **0.39/0.14/0.22/0.83** | **0.46/0.26/0.41/0.88** |
| T1-Initialized (w/o T2) | 0.32/0.10/0.17/0.83 | 0.42/0.16/0.24/0.84 | 0.39/0.14/0.22/0.83 | **0.46/0.26/0.41/0.88** |
| BART-Large-Base (w/o T1 & T2) | 0.36/0.12/0.17/0.83 | 0.38/0.13/0.22/0.83 | 0.38/0.13/0.21/0.83 | 0.44/0.24/0.39/0.88 |

**Note:** R1, R2, and RL denote ROUGE-1, ROUGE-2, and ROUGE-L, respectively. BERT-F1 denotes BERTScore F1.
