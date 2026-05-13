"""
main.py — Multilingual Text Classification Using Shared Semantic Representations

Full pipeline:
  Step 1: Data Collection (SIB-200 from HuggingFace)
  Step 2: Sampling (400 train / 100 test per language)
  Step 3: Preprocessing (Pipeline A — aggressive, Pipeline B — minimal)
  Step 4: Text Representation (TF-IDF, mBERT, XLM-R)
  Step 5: Models (LR + CNB on TF-IDF; LR on transformer embeddings)
  Step 6: Results & Analysis
         - Overall accuracy + macro-F1 (Direction 1)
         - Per-language breakdown (Gap 5)
         - Preprocessing ablation (Gap 3)
         - Confusion matrices
         - Error analysis (Gap 6)

Languages: English, Hindi, Tamil, Swahili
Dataset:   SIB-200 (Davlan/sib200)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel

from utils import (
    preprocess_aggressive,
    preprocess_minimal,
    get_transformer_embeddings,
    plot_grouped_bar_chart,
    plot_confusion_matrices,
    plot_ablation_chart,
    generate_error_analysis,
    LANG_DISPLAY,
)

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
LANGUAGES = ["eng_Latn", "hin_Deva", "tam_Taml", "swh_Latn"]
SAMPLE_PER_LANG_TRAIN = 400   # 4 × 400 = 1,600 training samples
SAMPLE_PER_LANG_TEST = 100    # 4 × 100 = 400 test samples
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STEP 1 — DATA COLLECTION
# ============================================================================
def load_sib200(languages: list) -> dict:
    """Load SIB-200 dataset for each language from HuggingFace."""
    print("=" * 70)
    print("STEP 1: DATA COLLECTION — Loading SIB-200 from HuggingFace")
    print("=" * 70)

    all_data = {}
    for lang in languages:
        print(f"  Loading {lang} ({LANG_DISPLAY.get(lang, lang)})...")
        ds = load_dataset("Davlan/sib200", lang)
        all_data[lang] = {
            "train": ds["train"].to_pandas(),
            "validation": ds["validation"].to_pandas(),
            "test": ds["test"].to_pandas(),
        }
        print(f"    Train: {len(all_data[lang]['train'])}, "
              f"Val: {len(all_data[lang]['validation'])}, "
              f"Test: {len(all_data[lang]['test'])}")
    return all_data


# ============================================================================
# STEP 2 — SAMPLING
# ============================================================================
def sample_data(all_data: dict, languages: list) -> tuple:
    """Sample fixed-size subsets per language and combine."""
    print("\n" + "=" * 70)
    print("STEP 2: SAMPLING — Creating balanced train/test splits")
    print("=" * 70)

    train_frames, test_frames = [], []

    for lang in languages:
        train_df = all_data[lang]["train"].sample(
            n=min(SAMPLE_PER_LANG_TRAIN, len(all_data[lang]["train"])),
            random_state=RANDOM_STATE,
        )
        test_df = all_data[lang]["test"].sample(
            n=min(SAMPLE_PER_LANG_TEST, len(all_data[lang]["test"])),
            random_state=RANDOM_STATE,
        )
        train_df["language"] = lang
        test_df["language"] = lang
        train_frames.append(train_df)
        test_frames.append(test_df)

    train_combined = pd.concat(train_frames, ignore_index=True)
    test_combined = pd.concat(test_frames, ignore_index=True)

    print(f"  Total train samples: {len(train_combined)}")
    print(f"  Total test samples:  {len(test_combined)}")
    print(f"  Label distribution (train):\n{train_combined['category'].value_counts().to_string()}")

    return train_combined, test_combined


# ============================================================================
# STEP 3 — PREPROCESSING (Gap 3 setup)
# ============================================================================
def apply_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Apply both Pipeline A (aggressive) and Pipeline B (minimal) to all data."""
    print("\n" + "=" * 70)
    print("STEP 3: PREPROCESSING — Applying Pipeline A and Pipeline B")
    print("=" * 70)

    # Pipeline A — Aggressive
    print("  Applying Pipeline A (aggressive cleaning)...")
    train_df["text_pipeline_A"] = train_df.apply(
        lambda row: preprocess_aggressive(row["text"], row["language"]), axis=1
    )
    test_df["text_pipeline_A"] = test_df.apply(
        lambda row: preprocess_aggressive(row["text"], row["language"]), axis=1
    )

    # Pipeline B — Minimal
    print("  Applying Pipeline B (minimal cleaning)...")
    train_df["text_pipeline_B"] = train_df["text"].apply(preprocess_minimal)
    test_df["text_pipeline_B"] = test_df["text"].apply(preprocess_minimal)

    # Show examples
    print("\n  Example (English):")
    eng_row = train_df[train_df["language"] == "eng_Latn"].iloc[0]
    print(f"    Original:   {eng_row['text'][:80]}...")
    print(f"    Pipeline A: {eng_row['text_pipeline_A'][:80]}...")
    print(f"    Pipeline B: {eng_row['text_pipeline_B'][:80]}...")

    return train_df, test_df


# ============================================================================
# STEP 4 — TEXT REPRESENTATION
# ============================================================================
def build_tfidf_features(train_df, test_df, pipeline_col: str) -> tuple:
    """Build TF-IDF features using character n-grams."""
    vectorizer = TfidfVectorizer(
        max_features=10000,
        analyzer="char_wb",
        ngram_range=(2, 4),
    )
    X_train = vectorizer.fit_transform(train_df[pipeline_col])
    X_test = vectorizer.transform(test_df[pipeline_col])
    return X_train, X_test, vectorizer


def build_transformer_features(
    train_df, test_df, pipeline_col: str, model_name: str
) -> tuple:
    """Extract [CLS] embeddings from a pretrained transformer."""
    print(f"    Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Auto-detect GPU
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"    Using device: {device}")

    print(f"    Extracting train embeddings ({len(train_df)} samples)...")
    t0 = time.time()
    X_train = get_transformer_embeddings(
        train_df[pipeline_col].tolist(), tokenizer, model,
        batch_size=16, max_length=128,
    )
    print(f"    Extracting test embeddings ({len(test_df)} samples)...")
    X_test = get_transformer_embeddings(
        test_df[pipeline_col].tolist(), tokenizer, model,
        batch_size=16, max_length=128,
    )
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s — shape: train={X_train.shape}, test={X_test.shape}")

    return X_train, X_test


def build_all_representations(train_df, test_df) -> dict:
    """
    Build all representations under both pipelines for ablation study.

    Returns a dict:
      {
        ("tfidf", "A"): (X_train, X_test),
        ("tfidf", "B"): (X_train, X_test),
        ("mbert", "A"): (X_train, X_test),
        ("mbert", "B"): (X_train, X_test),
        ("xlmr",  "A"): (X_train, X_test),
        ("xlmr",  "B"): (X_train, X_test),
      }
    """
    print("\n" + "=" * 70)
    print("STEP 4: TEXT REPRESENTATION — Building features for all models")
    print("=" * 70)

    reps = {}

    # --- TF-IDF under both pipelines ---
    print("\n  [TF-IDF] Pipeline A (aggressive):")
    X_tr, X_te, _ = build_tfidf_features(train_df, test_df, "text_pipeline_A")
    reps[("tfidf", "A")] = (X_tr, X_te)
    print(f"    Shape: train={X_tr.shape}, test={X_te.shape}")

    print("\n  [TF-IDF] Pipeline B (minimal):")
    X_tr, X_te, _ = build_tfidf_features(train_df, test_df, "text_pipeline_B")
    reps[("tfidf", "B")] = (X_tr, X_te)
    print(f"    Shape: train={X_tr.shape}, test={X_te.shape}")

    # --- mBERT under both pipelines ---
    print("\n  [mBERT] Pipeline B (minimal — primary):")
    X_tr, X_te = build_transformer_features(
        train_df, test_df, "text_pipeline_B", "bert-base-multilingual-cased"
    )
    reps[("mbert", "B")] = (X_tr, X_te)

    print("\n  [mBERT] Pipeline A (aggressive — ablation):")
    X_tr, X_te = build_transformer_features(
        train_df, test_df, "text_pipeline_A", "bert-base-multilingual-cased"
    )
    reps[("mbert", "A")] = (X_tr, X_te)

    # --- XLM-R under both pipelines ---
    print("\n  [XLM-R] Pipeline B (minimal — primary):")
    X_tr, X_te = build_transformer_features(
        train_df, test_df, "text_pipeline_B", "xlm-roberta-base"
    )
    reps[("xlmr", "B")] = (X_tr, X_te)

    print("\n  [XLM-R] Pipeline A (aggressive — ablation):")
    X_tr, X_te = build_transformer_features(
        train_df, test_df, "text_pipeline_A", "xlm-roberta-base"
    )
    reps[("xlmr", "A")] = (X_tr, X_te)

    return reps


# ============================================================================
# STEP 5 — MODELS
# ============================================================================
def train_all_models(reps: dict, y_train, y_test) -> dict:
    """
    Train classifiers on all representation × pipeline combinations.

    Returns a dict of { (model_name, pipeline): y_pred_test }
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODELS — Training classifiers")
    print("=" * 70)

    predictions = {}

    for pipeline_label in ["A", "B"]:
        print(f"\n  --- Pipeline {pipeline_label} ---")

        # TF-IDF + Logistic Regression
        X_tr, X_te = reps[("tfidf", pipeline_label)]
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
        lr.fit(X_tr, y_train)
        predictions[("TF-IDF + LR", pipeline_label)] = lr.predict(X_te)
        print(f"  ✓ TF-IDF + Logistic Regression")

        # TF-IDF + Complement Naive Bayes
        cnb = ComplementNB(alpha=0.1)
        cnb.fit(X_tr, y_train)
        predictions[("TF-IDF + CNB", pipeline_label)] = cnb.predict(X_te)
        print(f"  ✓ TF-IDF + Complement Naive Bayes")

        # mBERT + Logistic Regression
        X_tr, X_te = reps[("mbert", pipeline_label)]
        lr_mb = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
        lr_mb.fit(X_tr, y_train)
        predictions[("mBERT + LR", pipeline_label)] = lr_mb.predict(X_te)
        print(f"  ✓ mBERT + Logistic Regression")

        # XLM-R + Logistic Regression
        X_tr, X_te = reps[("xlmr", pipeline_label)]
        lr_xl = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
        lr_xl.fit(X_tr, y_train)
        predictions[("XLM-R + LR", pipeline_label)] = lr_xl.predict(X_te)
        print(f"  ✓ XLM-R + Logistic Regression")

    return predictions


# ============================================================================
# STEP 6 — RESULTS & ANALYSIS
# ============================================================================
def report_overall_results(predictions: dict, y_test) -> pd.DataFrame:
    """
    Table 1: Overall Accuracy and Macro-F1 for primary pipeline.
    TF-IDF uses Pipeline A, transformers use Pipeline B.
    """
    print("\n" + "=" * 70)
    print("STEP 6A: OVERALL RESULTS (Direction 1 — Primary Result)")
    print("=" * 70)

    # Primary configuration: TF-IDF → Pipeline A, transformers → Pipeline B
    primary_configs = [
        ("TF-IDF + LR",  "A"),
        ("TF-IDF + CNB", "A"),
        ("mBERT + LR",   "B"),
        ("XLM-R + LR",   "B"),
    ]

    rows = []
    for model_name, pipeline in primary_configs:
        y_pred = predictions[(model_name, pipeline)]
        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred, average="macro") * 100
        rows.append({
            "Model": model_name,
            "Pipeline": pipeline,
            "Accuracy (%)": round(acc, 2),
            "Macro-F1 (%)": round(f1, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTPUT_DIR, "overall_results.csv"), index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/overall_results.csv")
    return df


def report_per_language(predictions: dict, y_test, test_df, languages: list) -> pd.DataFrame:
    """
    Table 2: Per-language accuracy (Gap 5).
    """
    print("\n" + "=" * 70)
    print("STEP 6B: PER-LANGUAGE ACCURACY (Gap 5 — Language-wise Analysis)")
    print("=" * 70)

    primary_configs = [
        ("TF-IDF + LR",  "A"),
        ("TF-IDF + CNB", "A"),
        ("mBERT + LR",   "B"),
        ("XLM-R + LR",   "B"),
    ]

    lang_results = {}
    for model_name, pipeline in primary_configs:
        y_pred = predictions[(model_name, pipeline)]
        lang_results[model_name] = {}
        for lang in languages:
            mask = test_df["language"].values == lang
            acc = accuracy_score(y_test[mask], y_pred[mask]) * 100
            lang_results[model_name][lang] = round(acc, 2)

    # Build table
    rows = []
    for model_name in lang_results:
        row = {"Model": model_name}
        for lang in languages:
            row[LANG_DISPLAY[lang]] = lang_results[model_name][lang]
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUTPUT_DIR, "per_language_accuracy.csv"), index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/per_language_accuracy.csv")

    # Plot
    plot_grouped_bar_chart(
        data=lang_results,
        languages=languages,
        title="Per-Language Classification Accuracy: All Models",
        ylabel="Accuracy (%)",
        save_path=os.path.join(OUTPUT_DIR, "per_language_accuracy.png"),
    )

    return df


def report_ablation(predictions: dict, y_test) -> pd.DataFrame:
    """
    Table 3: Preprocessing ablation — Pipeline A vs Pipeline B (Gap 3).
    """
    print("\n" + "=" * 70)
    print("STEP 6C: PREPROCESSING ABLATION (Gap 3)")
    print("=" * 70)

    model_names = ["TF-IDF + LR", "TF-IDF + CNB", "mBERT + LR", "XLM-R + LR"]
    rows = []
    for model_name in model_names:
        acc_a = accuracy_score(y_test, predictions[(model_name, "A")]) * 100
        acc_b = accuracy_score(y_test, predictions[(model_name, "B")]) * 100
        rows.append({
            "Model": model_name,
            "Pipeline A (Aggressive)": round(acc_a, 2),
            "Pipeline B (Minimal)": round(acc_b, 2),
            "Δ (B − A)": round(acc_b - acc_a, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"), index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR}/ablation_results.csv")

    # Plot
    plot_ablation_chart(df, os.path.join(OUTPUT_DIR, "ablation_comparison.png"))

    return df


def report_confusion_matrices(predictions, y_test, test_df, le, languages):
    """
    Confusion matrices for best model (XLM-R) per language.
    """
    print("\n" + "=" * 70)
    print("STEP 6D: CONFUSION MATRICES (XLM-R + LR, Pipeline B)")
    print("=" * 70)

    best_pred = predictions[("XLM-R + LR", "B")]
    plot_confusion_matrices(
        y_true=y_test,
        y_pred=best_pred,
        test_df=test_df,
        languages=languages,
        label_names=le.classes_.tolist(),
        model_name="XLM-R + LR",
        save_dir=OUTPUT_DIR,
    )


def report_error_analysis(predictions, y_test, test_df, le, languages):
    """
    Qualitative error analysis table (Gap 6).
    """
    print("\n" + "=" * 70)
    print("STEP 6E: ERROR ANALYSIS (Gap 6 — Qualitative)")
    print("=" * 70)

    best_pred = predictions[("XLM-R + LR", "B")]
    error_df = generate_error_analysis(
        y_true=y_test,
        y_pred=best_pred,
        test_df=test_df,
        label_encoder=le,
        languages=languages,
        n_per_lang=3,
    )

    if len(error_df) > 0:
        # Truncate long text for display
        error_df["Text"] = error_df["Text"].apply(
            lambda x: x[:80] + "..." if len(x) > 80 else x
        )
        print(error_df.to_string(index=False))
        error_df.to_csv(os.path.join(OUTPUT_DIR, "error_analysis.csv"), index=False)
        print(f"  ✓ Saved: {OUTPUT_DIR}/error_analysis.csv")
    else:
        print("  No misclassifications found (unlikely but possible).")

    return error_df


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Run the full pipeline end-to-end."""
    start_time = time.time()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  Multilingual Text Classification Using Shared Semantic          ║")
    print("║  Representations — Full Pipeline                                 ║")
    print("║  Dataset: SIB-200 | Models: TF-IDF, mBERT, XLM-R               ║")
    print("║  Languages: English, Hindi, Tamil, Swahili                       ║")
    print("╚" + "═" * 68 + "╝\n")

    # Step 1: Load data
    all_data = load_sib200(LANGUAGES)

    # Step 2: Sample
    train_df, test_df = sample_data(all_data, LANGUAGES)

    # Step 3: Preprocess
    train_df, test_df = apply_preprocessing(train_df, test_df)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["category"])
    y_test = le.transform(test_df["category"])
    print(f"\n  Label classes: {le.classes_.tolist()}")

    # Step 4: Build representations (all model × pipeline combinations)
    reps = build_all_representations(train_df, test_df)

    # Step 5: Train models
    predictions = train_all_models(reps, y_train, y_test)

    # Step 6: Results & Analysis
    overall_df = report_overall_results(predictions, y_test)
    per_lang_df = report_per_language(predictions, y_test, test_df, LANGUAGES)
    ablation_df = report_ablation(predictions, y_test)
    report_confusion_matrices(predictions, y_test, test_df, le, LANGUAGES)
    error_df = report_error_analysis(predictions, y_test, test_df, le, LANGUAGES)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — Total time: {elapsed / 60:.1f} minutes")
    print("=" * 70)
    print(f"\n  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    • {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
