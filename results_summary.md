# Results Summary
**Multilingual Text Classification Using Shared Semantic Representations**

**Dataset:** SIB-200 · 4 languages · 7 classes · 1,600 train / 400 test samples

---

## Overall Model Comparison

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LR | 53.50% | 37.46% |
| TF-IDF + CNB | 64.25% | 59.76% |
| mBERT + LR | 69.50% | 65.51% |
| **XLM-R + LR** | **80.75%** | **77.17%** |

> XLM-R outperforms the best TF-IDF baseline by **+16.5% accuracy** and **+17.4% Macro-F1**.

## Per-Language Accuracy

| Model | English | Hindi | Tamil | Swahili |
|---|---|---|---|---|
| TF-IDF + LR | 53% | 54% | 55% | 52% |
| TF-IDF + CNB | 63% | 54% | 69% | 71% |
| mBERT + LR | 82% | 71% | 65% | 60% |
| **XLM-R + LR** | **86%** | **82%** | **81%** | **74%** |

![Per-language accuracy chart](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/per_language_accuracy.png)

- XLM-R is the most consistent (86% → 74%, range = 12 pp)
- mBERT drops steeply on low-resource languages (82% → 60%, range = 22 pp)
- **Surprise:** TF-IDF + CNB beats mBERT on Tamil (69 vs 65%) and Swahili (71 vs 60%)

## Preprocessing Ablation

| Model | Aggressive | Minimal | Δ |
|---|---|---|---|
| TF-IDF + CNB | 64.25% | 62.00% | **−2.25** (aggressive helps) |
| mBERT + LR | 62.00% | 69.50% | **+7.50** (aggressive hurts) |
| XLM-R + LR | 77.00% | 80.75% | **+3.75** (aggressive hurts) |

![Ablation chart](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/ablation_comparison.png)

> Aggressive preprocessing helps TF-IDF but hurts transformers — the same pipeline cannot serve both.

## Per-Class F1 (Weakest → Strongest)

| Category | TF-IDF+LR | TF-IDF+CNB | mBERT+LR | XLM-R+LR |
|---|---|---|---|---|
| entertainment | 0.0 | 33.3 | 55.3 | 58.5 |
| geography | 11.1 | 68.7 | 59.3 | 87.1 |
| health | 32.7 | 45.7 | 50.7 | 57.6 |
| sports | 32.4 | 64.0 | 70.4 | 80.0 |
| science/tech ⬅ majority | 60.5 | 65.7 | 76.7 | 82.2 |
| politics | 60.9 | 74.3 | 75.4 | 92.3 |
| travel | 64.7 | 66.7 | 70.8 | 82.5 |

![Per-class F1 heatmap](/Users/sukesh/.gemini/antigravity/brain/8db580e2-94ad-4377-b822-e584e5c43978/per_class_f1_heatmap.png)

> Entertainment and health are the weakest categories across all models. Science/technology is inflated by 2.7× overrepresentation in training data.

---

## Key Takeaways

1. **Shared representations win** — XLM-R (80.75%) ≫ TF-IDF (64.25%)
2. **Aggregate metrics are misleading** — XLM-R varies 12 pp across languages
3. **Preprocessing matters** — wrong pipeline costs mBERT 7.5 pp
4. **Don't dismiss classical models** — TF-IDF+CNB beats mBERT on agglutinative languages
