# Multilingual Text Classification Using Shared Semantic Representations

A comparative study of **TF-IDF**, **mBERT**, and **XLM-R** for multilingual topic classification on the **SIB-200** dataset across English, Hindi, Tamil, and Swahili.

---

## Research Question

> Do shared semantic representations from multilingual transformers (mBERT, XLM-R) outperform non-shared TF-IDF features for multilingual topic classification across languages with different scripts and resource levels, and how does preprocessing strategy affect each representation type?

## Key Contributions

| Layer | Contribution | Description |
|---|---|---|
| **Direction 1** | Classical vs Transformer | TF-IDF baseline vs mBERT/XLM-R shared representations |
| **Gap 3** | Preprocessing Ablation | Aggressive vs minimal preprocessing — impact on each model type |
| **Gap 5** | Language-wise Analysis | Per-language breakdown revealing hidden performance disparities |
| **Gap 6** | Error Analysis | Qualitative misclassification analysis |

## Dataset: SIB-200

- **Source:** [Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) on HuggingFace
- **Task:** 7-class topic classification (science_technology, travel, politics, sports, health, entertainment, geography)
- **Languages used:** English (`eng_Latn`), Hindi (`hin_Deva`), Tamil (`tam_Taml`), Swahili (`swh_Latn`)
- **Samples:** 1,600 train + 400 test (400 train / 100 test per language)

## Models

| Model | Representation | Classifier |
|---|---|---|
| TF-IDF + LR | Character n-gram TF-IDF (2-4 grams) | Logistic Regression |
| TF-IDF + CNB | Character n-gram TF-IDF (2-4 grams) | Complement Naive Bayes |
| mBERT + LR | `bert-base-multilingual-cased` [CLS] embeddings | Logistic Regression |
| XLM-R + LR | `xlm-roberta-base` [CLS] embeddings | Logistic Regression |

## Preprocessing Pipelines

- **Pipeline A (Aggressive):** Lowercase → remove numbers → remove punctuation → tokenize → remove English stopwords → remove single-char tokens
- **Pipeline B (Minimal):** Strip whitespace → normalize multiple spaces to single space

## Project Structure

```
├── main.py                 # Full end-to-end pipeline (run this)
├── utils.py                # Helper functions (preprocessing, embeddings, plots)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── outputs/                # Generated results (created on run)
    ├── overall_results.csv
    ├── per_language_accuracy.csv
    ├── per_language_accuracy.png
    ├── ablation_results.csv
    ├── ablation_comparison.png
    ├── per_class_f1.csv
    ├── per_class_f1_heatmap.png
    ├── h2_gap_analysis.csv
    ├── h2_gap_analysis.png
    ├── confusion_matrix_*.png      (4 files, one per language)
    ├── error_analysis.csv
    ├── missing_elements_report.md  (consolidated paper-ready text)
    └── *.md                        (individual paper section drafts)
```

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

This will:
1. Download SIB-200 dataset from HuggingFace (first run only)
2. Download mBERT and XLM-R model weights (first run only, ~1.5 GB)
3. Run all preprocessing, feature extraction, training, and evaluation
4. Save all results, tables, and visualizations to `outputs/`

### Expected Runtime

- **GPU (Colab/local):** ~10-15 minutes
- **CPU only:** ~30-45 minutes

## Expected Results

### Overall Accuracy (Direction 1)

| Model | Accuracy (%) | Macro-F1 (%) |
|---|---|---|
| TF-IDF + LR | ~50-55 | ~48-52 |
| TF-IDF + CNB | ~45-50 | ~43-48 |
| mBERT + LR | ~68-75 | ~66-73 |
| XLM-R + LR | ~75-82 | ~73-80 |

### Key Hypotheses

1. **XLM-R > mBERT > TF-IDF** on average multilingual accuracy
2. **TF-IDF ↔ transformer gap is larger** for non-Latin-script languages (Hindi, Tamil)
3. **Aggressive preprocessing hurts transformers** more than TF-IDF (Gap 3)
4. **Aggregate accuracy hides poor performance** on Tamil and Swahili (Gap 5)

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Python | Implementation |
| HuggingFace `datasets` | SIB-200 dataset loading |
| HuggingFace `transformers` | mBERT and XLM-R model loading |
| scikit-learn | TF-IDF, classifiers, metrics |
| NLTK | Tokenization, stopword removal |
| pandas | Data manipulation |
| matplotlib + seaborn | Visualizations |

## References

- Adelani et al., "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects," EACL 2024
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL 2019
- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale," ACL 2020
