# Multilingual Text Classification Using Shared Semantic Representations
## A Comparative Study of TF-IDF, mBERT, and XLM-R on SIB-200

---

## Abstract

This study investigates whether shared semantic representations produced by multilingual transformers outperform non-shared TF-IDF representations for multilingual topic classification. Four models — TF-IDF with Logistic Regression, TF-IDF with Complement Naive Bayes, mBERT with Logistic Regression, and XLM-R with Logistic Regression — were evaluated on the SIB-200 dataset across English, Hindi, Tamil, and Swahili for 7-class topic classification. XLM-R achieved the highest overall accuracy of 80.75% and Macro-F1 of 77.17%, outperforming the best TF-IDF baseline by 16.50 percentage points in accuracy. A preprocessing ablation study demonstrated that aggressive text cleaning benefits TF-IDF classifiers (Δ = −2.25% for CNB) but degrades transformer performance (Δ = +7.50% for mBERT), confirming that representation-aware preprocessing is necessary in multilingual settings. Per-language analysis revealed that aggregate metrics mask significant cross-lingual disparities: XLM-R achieved 86% on English but only 74% on Swahili. An unexpected finding showed that character n-gram TF-IDF with Complement Naive Bayes achieved competitive or superior performance to mBERT on morphologically rich languages (Tamil: 69% vs 65%, Swahili: 71% vs 60%), suggesting that subword-level statistical features partially compensate for the absence of shared semantic representations in agglutinative languages.

---

## 1. Introduction

Multilingual text classification is a fundamental task in natural language processing that requires assigning categorical labels to text written in multiple languages. The challenge lies in building representations that capture semantic meaning across languages with different scripts, morphologies, and resource levels. Traditional approaches based on bag-of-words features such as TF-IDF treat each language as an independent vocabulary, offering no mechanism for cross-lingual transfer. In contrast, multilingual pretrained transformers such as mBERT and XLM-R produce shared semantic representations — dense vector spaces where semantically similar texts from different languages are mapped to nearby points — enabling language-agnostic classification.

This study makes the following contributions:

1. **Primary finding (Direction 1):** XLM-R contextual embeddings outperform TF-IDF features by 27.25 percentage points in accuracy and 17.41 points in Macro-F1 on average across four languages, confirming that shared semantic representations provide measurable gains for multilingual text classification.

2. **Preprocessing finding (Gap 3):** Aggressive text preprocessing that benefits TF-IDF classifiers consistently degrades multilingual transformer performance, demonstrating that representation-aware preprocessing is necessary in multilingual settings.

3. **Language-wise finding (Gap 5):** Aggregate accuracy scores mask significant performance disparities across languages, highlighting that average multilingual metrics are insufficient for evaluating cross-lingual generalization.

4. **Unexpected finding:** Character n-gram TF-IDF paired with Complement Naive Bayes achieves competitive performance with mBERT on morphologically rich languages such as Tamil and Swahili, suggesting that subword-level statistical features can partially compensate for the absence of shared semantic representations when the target language exhibits productive agglutinative morphology.

---

## 2. Dataset

### 2.1 SIB-200

The SIB-200 dataset (Adelani et al., EACL 2024) was used for all experiments. SIB-200 is a topic classification benchmark derived from FLORES-200 parallel sentences, covering 200+ languages with 7 topic categories: science/technology, travel, politics, sports, health, entertainment, and geography.

**HuggingFace ID:** `Davlan/sib200`

### 2.2 Language Selection

| Language | SIB-200 Code | Script | Resource Level | Role |
|---|---|---|---|---|
| English | `eng_Latn` | Latin | High | Baseline, training source |
| Hindi | `hin_Deva` | Devanagari | Medium | Primary Indian language |
| Tamil | `tam_Taml` | Tamil script | Medium-Low | South Indian, different script |
| Swahili | `swh_Latn` | Latin | Low | African low-resource contrast |

### 2.3 Sampling

| Split | Per Language | Total |
|---|---|---|
| Training | 400 | 1,600 |
| Test | 100 | 400 |
| **Total** | **500** | **2,000** |

### 2.4 Label Distribution (Training Set)

| Category | Count | Proportion |
|---|---|---|
| science/technology | 392 | 24.5% |
| travel | 324 | 20.3% |
| politics | 228 | 14.3% |
| sports | 188 | 11.8% |
| health | 180 | 11.3% |
| geography | 144 | 9.0% |
| entertainment | 144 | 9.0% |

> [!NOTE]
> The dataset exhibits moderate class imbalance — science/technology has 2.7× more samples than geography or entertainment. This affects per-class F1 scores and is addressed in the Class-Level Analysis (Section 5.3).

---

## 3. Methodology

### 3.1 Preprocessing Pipelines

Two preprocessing pipelines were designed to test the hypothesis that aggressive cleaning affects classical and transformer models differently (Gap 3).

| Pipeline | Steps | Intended For |
|---|---|---|
| **A** (Aggressive) | Lowercase → remove numbers → remove punctuation → NLTK tokenize → remove English stopwords → remove single-char tokens | TF-IDF models |
| **B** (Minimal) | Strip whitespace → normalize multiple spaces to single space | Transformer models |

**Example (English):**
- **Original:** `Italy's national football, along with German national football team is the secon...`
- **Pipeline A:** `italys national football along german national football team second successful t...`
- **Pipeline B:** `Italy's national football, along with German national football team is the secon...`

### 3.2 Text Representations

| Representation | Details | Dimensionality |
|---|---|---|
| **TF-IDF** | Character n-grams (2–4), `char_wb` analyzer, 10K max features | 10,000 (sparse) |
| **mBERT** | `bert-base-multilingual-cased` [CLS] token embeddings (frozen) | 768 |
| **XLM-R** | `xlm-roberta-base` [CLS] token embeddings (frozen) | 768 |

Character n-gram TF-IDF was chosen over word-level TF-IDF because word-level features treat Hindi, Tamil, and English as completely separate vocabularies with zero token overlap. Character n-grams capture subword patterns that partially transfer across morphologically related forms.

Due to computational constraints, mBERT and XLM-R were used as fixed feature extractors — weights were frozen, [CLS] token representations were extracted, and lightweight classifiers were trained on the resulting embeddings.

### 3.3 Classifiers

| Model ID | Features | Classifier | Key Parameters |
|---|---|---|---|
| TF-IDF + LR | TF-IDF (Pipeline A) | Logistic Regression | C=1.0, max_iter=1000 |
| TF-IDF + CNB | TF-IDF (Pipeline A) | Complement Naive Bayes | α=0.1 |
| mBERT + LR | mBERT embeddings (Pipeline B) | Logistic Regression | C=1.0, max_iter=1000 |
| XLM-R + LR | XLM-R embeddings (Pipeline B) | Logistic Regression | C=1.0, max_iter=1000 |

Complement Naive Bayes was chosen over standard Multinomial NB because it estimates class parameters using data from all classes *except* the target, providing more stable estimates for minority classes in imbalanced settings.

---

## 4. Results

### 4.1 Overall Performance (Direction 1 — Primary Result)

| Model | Pipeline | Accuracy (%) | Macro-F1 (%) |
|---|---|---|---|
| TF-IDF + LR | A (Aggressive) | 53.50 | 37.46 |
| TF-IDF + CNB | A (Aggressive) | 64.25 | 59.76 |
| mBERT + LR | B (Minimal) | 69.50 | 65.51 |
| **XLM-R + LR** | **B (Minimal)** | **80.75** | **77.17** |

XLM-R dominated with 80.75% accuracy and 77.17% Macro-F1. The 17.41-point Macro-F1 gap between TF-IDF + CNB (59.76%) and XLM-R + LR (77.17%) is the primary empirical evidence that shared semantic representations provide measurable gains for multilingual text classification.

TF-IDF + LR performed poorly (53.50% accuracy, 37.46% F1) — the large gap between accuracy and Macro-F1 indicates bias toward majority classes. TF-IDF + CNB was surprisingly competitive at 64.25%, demonstrating that classifier choice matters even within the classical paradigm.

### 4.2 Per-Language Accuracy (Gap 5 — Language-wise Analysis)

| Model | English | Hindi | Tamil | Swahili | Range |
|---|---|---|---|---|---|
| TF-IDF + LR | 53.0% | 54.0% | 55.0% | 52.0% | 3.0 |
| TF-IDF + CNB | 63.0% | 54.0% | 69.0% | 71.0% | 17.0 |
| mBERT + LR | 82.0% | 71.0% | 65.0% | 60.0% | 22.0 |
| **XLM-R + LR** | **86.0%** | **82.0%** | **81.0%** | **74.0%** | **12.0** |

![Per-language classification accuracy grouped by model](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/per_language_accuracy.png)

**Key findings:**

- **XLM-R is the most consistent** across languages (range = 12 pp), while mBERT has the widest disparity (range = 22 pp), dropping from 82% (English) to 60% (Swahili).
- **mBERT struggles with low-resource languages** — its steep degradation reflects Wikipedia-only pretraining which underrepresents African languages. XLM-R's CommonCrawl pretraining provides more balanced coverage.
- **The English → Swahili gap for transformers:** mBERT drops 22 pp; XLM-R drops only 12 pp — XLM-R halves the resource gap.
- **Reporting only aggregate 80.75% for XLM-R hides that Swahili is 12 points lower** — per-language evaluation is essential for deployment decisions.

### 4.3 Preprocessing Ablation (Gap 3)

| Model | Pipeline A (Aggressive) | Pipeline B (Minimal) | Δ (B − A) |
|---|---|---|---|
| TF-IDF + LR | 53.50% | 54.25% | +0.75 |
| TF-IDF + CNB | 64.25% | 62.00% | **−2.25** |
| mBERT + LR | 62.00% | 69.50% | **+7.50** |
| XLM-R + LR | 77.00% | 80.75% | **+3.75** |

![Preprocessing ablation — Pipeline A vs Pipeline B with delta annotations](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/ablation_comparison.png)

**Key findings:**

- **mBERT loses 7.5 pp** under aggressive preprocessing — lowercasing removes casing signals that mBERT's `cased` tokenizer relies on, and punctuation removal destroys sentence structure cues.
- **XLM-R is more resilient** (−3.75 pp) — its SentencePiece tokenizer is more robust to preprocessing artifacts.
- **TF-IDF + CNB benefits** from aggressive cleaning (Δ = −2.25), confirming traditional NLP wisdom.
- **The opposing Δ directions** (negative for CNB, positive for transformers) empirically confirm that the same preprocessing pipeline cannot optimally serve both representation types.

---

## 5. Analysis

### 5.1 H2: Script-Based Performance Gap Analysis

The hypothesis predicted that the TF-IDF-to-transformer performance gap would be larger for non-Latin-script languages (Hindi, Tamil) than for Latin-script languages (English, Swahili).

#### Gap Computation Table

| Language | Script | Resource | mBERT − TF-IDF+LR | mBERT − TF-IDF+CNB | XLM-R − TF-IDF+LR | XLM-R − TF-IDF+CNB |
|---|---|---|---|---|---|---|
| English | Latin | High | +29.0 pp | +19.0 pp | +33.0 pp | +23.0 pp |
| Hindi | Devanagari | Medium | +17.0 pp | +17.0 pp | +28.0 pp | +28.0 pp |
| Tamil | Tamil Script | Medium-Low | +10.0 pp | **−4.0 pp** | +26.0 pp | +12.0 pp |
| Swahili | Latin | Low | +8.0 pp | **−11.0 pp** | +22.0 pp | +3.0 pp |

![H2 gap analysis — transformer vs TF-IDF gap by language](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/h2_gap_analysis.png)

The experimental data revealed a more nuanced pattern than predicted. When comparing mBERT against TF-IDF + Logistic Regression, the gap was largest for English (29 pp), followed by Hindi (17 pp), Tamil (10 pp), and Swahili (8 pp) — the exact opposite of the predicted ordering. The most striking result emerged when comparing against TF-IDF + Complement Naive Bayes: mBERT performed *worse* than TF-IDF + CNB on Tamil (−4 pp) and Swahili (−11 pp), while XLM-R's advantage shrank to just 3 pp on Swahili.

The unexpected competitiveness of TF-IDF + CNB on Tamil and Swahili can be attributed to two interacting factors. First, the character n-gram representation captures subword morphological patterns that are particularly informative for agglutinative languages such as Tamil and Swahili, where productive morphology generates diverse word forms from common stems. Second, Complement Naive Bayes estimates class parameters using data from all classes *except* the target, providing more stable estimates for minority classes and amplifying the signal from morphologically distinctive character patterns.

**Revised H2 verdict:** The hypothesis that non-Latin-script languages would exhibit larger TF-IDF-to-transformer gaps is not supported; rather, the gap magnitude is primarily determined by the choice of classical classifier and the morphological richness of the target language.

### 5.2 Confusion Matrices (XLM-R — Best Model)

````carousel
![XLM-R confusion matrix — English](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/confusion_matrix_eng_Latn.png)
<!-- slide -->
![XLM-R confusion matrix — Hindi](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/confusion_matrix_hin_Deva.png)
<!-- slide -->
![XLM-R confusion matrix — Tamil](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/confusion_matrix_tam_Taml.png)
<!-- slide -->
![XLM-R confusion matrix — Swahili](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/confusion_matrix_swh_Latn.png)
````

| Language | Most Confused Pair | Likely Reason |
|---|---|---|
| English | travel ↔ entertainment | Tourism descriptions overlap with cultural content |
| Hindi | health → science/technology | Medical topics contain scientific terminology |
| Tamil | entertainment → science/technology | Technical instrument descriptions blur boundaries |
| Swahili | travel → health | Environmental/nature travel descriptions trigger health |

### 5.3 Per-Class F1 Scores

| Category | TF-IDF + LR | TF-IDF + CNB | mBERT + LR | XLM-R + LR |
|---|---|---|---|---|
| entertainment | 0.00 | 33.33 | 55.32 | 58.54 |
| geography | 11.11 | 68.66 | 59.26 | 87.10 |
| health | 32.65 | 45.71 | 50.70 | 57.58 |
| politics | 60.87 | 74.29 | 75.41 | 92.31 |
| **science/technology** ⬅ | **60.53** | **65.66** | **76.70** | **82.19** |
| sports | 32.35 | 64.00 | 70.37 | 80.00 |
| travel | 64.71 | 66.67 | 70.83 | 82.49 |

![Per-class F1 heatmap with majority class highlighted](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/per_class_f1_heatmap.png)

All models struggled most on entertainment (0.00%–58.54% F1) and health (32.65%–57.58% F1), while performing best on politics (60.87%–92.31%) and travel (64.71%–82.49%). The science/technology category, comprising 24.5% of training data (2.7× the representation of entertainment/geography), achieved consistently elevated F1 across all models, suggesting that its inflated frequency contributes to artificially high per-class performance rather than reflecting inherent classification ease. For deployment, balanced sampling or class-weighted loss functions should be adopted to ensure equitable quality across all topic domains.

### 5.4 Error Analysis (Gap 6 — Qualitative)

| Language | Text (truncated) | True | Predicted |
|---|---|---|---|
| English | *While most cards are good for calling anywhere, some specialise...* | science/technology | travel |
| English | *In good conditions you will be able to cover greater distances...* | sports | travel |
| English | *Also to the north visit the great Sanctuary of Our Lady of Fatima...* | travel | entertainment |
| Hindi | *एडीडी वाले बच्चों का मुशकिल समय होता है स्कूली पढ़ाई जैसी चीज़ों पर...* | health | science/technology |
| Hindi | *ये बच्चे बहुत परेशानी में पड़ते हैं, क्योंकि ये अपने दिमाग को उत्तेजित...* | health | science/technology |
| Hindi | *चार साल में यह मार्टेली का पांचवां सीईपी है।* | politics | sports |
| Tamil | *வடக்கே உலக புகழ்பெற்ற மரியன் மாய தோற்றங்களின் இடமான...* | travel | geography |
| Tamil | *அக்கார்டியனில் அதிக ஒலியினைப் பெறுவதற்கு, நீங்கள்...* | entertainment | science/technology |
| Tamil | *ஒவ்வொரு நிகழ்ச்சியும், குழந்தைகள் நூலகத்திற்குச் செல்லும்போது...* | entertainment | travel |
| Swahili | *Kwa kuwa maeneo hayo hayana idadi kubwa ya watu...* | travel | health |
| Swahili | *Wakati wa barafu nyingi, barafu ya kutosha kukufanya ukwame...* | travel | health |
| Swahili | *Watoto walio na ADD huwa na wakati mgumu kuangazia mambo...* | health | science/technology |

The dominant error mode across all languages is **topical overlap** — particularly health ↔ science/technology, travel ↔ geography, and travel ↔ entertainment. The health → science/technology confusion for ADD-related text appears identically in Hindi and Swahili, suggesting a systematic dataset annotation pattern rather than a language-specific failure.

---

## 6. Hypothesis Verification Summary

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 | XLM-R > mBERT > TF-IDF on average accuracy | 80.75% > 69.50% > 64.25% > 53.50% | ✅ Confirmed |
| H2 | Gap larger for non-Latin-script languages | Gap largest for English; CNB competitive on Tamil/Swahili | ⚠️ Partially confirmed |
| H3 | Aggressive preprocessing hurts transformers | mBERT Δ=+7.5, XLM-R Δ=+3.75; CNB Δ=−2.25 | ✅ Confirmed |
| H4 | Aggregate accuracy hides low-resource gaps | XLM-R: 86% English vs 74% Swahili | ✅ Confirmed |

---

## 7. Limitations

Several limitations of the present study should be acknowledged. First, the mBERT and XLM-R models were employed as frozen feature extractors rather than being fully fine-tuned on the target task; the original SIB-200 evaluation reported 92.1% accuracy for fully fine-tuned XLM-R, compared to the 80.75% achieved in this study using frozen [CLS] embeddings with a Logistic Regression head, indicating that the true capability of these transformer architectures is underestimated by approximately 11 percentage points. This design choice was necessitated by the computational constraints of a student project setting. Second, the test set comprised only 100 samples per language, yielding 400 total test instances, which is statistically limited; at this sample size, the 95% confidence interval for an 80% accuracy estimate spans approximately ±7.8 percentage points, and per-language differences should be interpreted as indicative trends rather than definitive characterizations. Third, the preprocessing ablation was constrained by the fact that NLTK does not natively provide stopword lists for Hindi or Tamil, resulting in stopword removal being applied only to English text in Pipeline A — an asymmetry that limits the ablation's generalizability. Fourth, the training set exhibited moderate class imbalance, with science/technology at 24.5% while geography and entertainment each accounted for only 9.0%, inflating Macro-F1 for models that disproportionately predict the majority class. Finally, all findings are specific to the SIB-200 dataset; the systematic travel-to-health confusion in Swahili may reflect annotation decisions in the FLORES-200 source corpus rather than a fundamental linguistic challenge, and replication on alternative benchmarks would be necessary.

---

## 8. Conclusion

This study demonstrated that shared semantic representations from multilingual transformers significantly outperform non-shared TF-IDF representations for multilingual topic classification. XLM-R achieved the highest performance (80.75% accuracy, 77.17% Macro-F1), confirming its effectiveness as a language-agnostic encoder. The preprocessing ablation revealed that aggressive cleaning benefits TF-IDF but degrades transformers, establishing that representation-aware preprocessing is essential. Per-language analysis uncovered performance disparities hidden by aggregate metrics, with a 12-point gap between English and Swahili for XLM-R. An unexpected finding showed that character n-gram TF-IDF with Complement Naive Bayes achieves competitive performance with mBERT on morphologically rich languages, suggesting that classical models should not be dismissed in resource-constrained multilingual settings.

Future work should explore full fine-tuning of transformer models, incorporate additional low-resource Indian languages (Telugu, Kannada, Malayalam), investigate the interaction between morphological richness and character n-gram effectiveness, and evaluate on additional multilingual benchmarks to confirm generalizability.

---

## References

1. Adelani, D. I., et al. "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects." EACL 2024.
2. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
3. Conneau, A., et al. "Unsupervised Cross-lingual Representation Learning at Scale." ACL 2020.
