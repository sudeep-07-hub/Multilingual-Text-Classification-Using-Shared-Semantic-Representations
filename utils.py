"""
utils.py — Helper functions for Multilingual Text Classification pipeline.

Contains:
- Preprocessing pipelines (aggressive + minimal)
- Generic transformer embedding extractor
- Visualization utilities (bar charts, confusion matrices)
- Error analysis table generator
"""

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ENGLISH_STOPWORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Language display names (used in plots / tables)
# ---------------------------------------------------------------------------
LANG_DISPLAY = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "tam_Taml": "Tamil",
    "swh_Latn": "Swahili",
}


# ============================= PREPROCESSING ===============================

def preprocess_aggressive(text: str, language: str = "eng_Latn") -> str:
    """
    Pipeline A — Traditional aggressive NLP cleaning:
    - Lowercase
    - Remove numbers
    - Remove punctuation
    - Tokenize (NLTK word_tokenize)
    - Remove English stopwords (English text only)
    - Remove single-character tokens
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)                                     # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))     # punctuation
    tokens = word_tokenize(text)
    if language == "eng_Latn":
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    tokens = [t for t in tokens if len(t) > 1]                          # single chars
    return " ".join(tokens)


def preprocess_minimal(text: str) -> str:
    """
    Pipeline B — Minimal cleaning for transformer models:
    - Strip leading/trailing whitespace
    - Normalize multiple spaces to a single space
    - Keep punctuation, casing, and script markers intact
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ======================== TRANSFORMER EMBEDDINGS ===========================

def get_transformer_embeddings(
    texts: list,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract [CLS] token embeddings from a HuggingFace transformer model.

    Works for both mBERT and XLM-R (or any AutoModel).
    Returns an (N, hidden_dim) numpy array.
    """
    import torch

    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Move to same device as model
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # [CLS] is the first token (index 0)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)


# ============================== VISUALIZATIONS =============================

def plot_grouped_bar_chart(
    data: dict,
    languages: list,
    title: str,
    ylabel: str,
    save_path: str,
):
    """
    Plot a grouped bar chart with one group per language and one bar per model.

    Parameters
    ----------
    data : dict
        {model_name: {lang_code: value, ...}, ...}
    languages : list
        Ordered list of language codes.
    title, ylabel, save_path : str
    """
    model_names = list(data.keys())
    x = np.arange(len(languages))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("viridis", len(model_names))

    for i, model_name in enumerate(model_names):
        vals = [data[model_name].get(lang, 0) for lang in languages]
        bars = ax.bar(x + i * width, vals, width, label=model_name, color=colors[i])
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([LANG_DISPLAY.get(l, l) for l in languages])
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    languages: list,
    label_names: list,
    model_name: str,
    save_dir: str,
):
    """
    Generate and save one confusion matrix per language for the given model.
    """
    import os

    for lang in languages:
        lang_mask = test_df["language"].values == lang
        y_true_lang = y_true[lang_mask]
        y_pred_lang = y_pred[lang_mask]

        cm = confusion_matrix(y_true_lang, y_pred_lang, labels=range(len(label_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

        fig, ax = plt.subplots(figsize=(8, 7))
        disp.plot(ax=ax, colorbar=False, xticks_rotation=45, cmap="Blues")
        display_name = LANG_DISPLAY.get(lang, lang)
        ax.set_title(
            f"{model_name} Confusion Matrix — {display_name}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        fname = os.path.join(save_dir, f"confusion_matrix_{lang}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {fname}")


def plot_ablation_chart(ablation_df: pd.DataFrame, save_path: str):
    """
    Horizontal bar chart showing Pipeline A vs Pipeline B accuracy per model,
    with delta annotations.
    """
    models = ablation_df["Model"].tolist()
    y_pos = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_height = 0.35

    bars_a = ax.barh(
        y_pos - bar_height / 2,
        ablation_df["Pipeline A (Aggressive)"],
        bar_height,
        label="Pipeline A (Aggressive)",
        color=sns.color_palette("Set2")[0],
    )
    bars_b = ax.barh(
        y_pos + bar_height / 2,
        ablation_df["Pipeline B (Minimal)"],
        bar_height,
        label="Pipeline B (Minimal)",
        color=sns.color_palette("Set2")[1],
    )

    # Delta annotations
    for i, (a_val, b_val, delta) in enumerate(
        zip(
            ablation_df["Pipeline A (Aggressive)"],
            ablation_df["Pipeline B (Minimal)"],
            ablation_df["Δ (B − A)"],
        )
    ):
        max_val = max(a_val, b_val)
        color = "green" if delta > 0 else ("red" if delta < 0 else "gray")
        ax.text(
            max_val + 1, i, f"Δ = {delta:+.2f}%",
            va="center", fontsize=10, fontweight="bold", color=color,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        "Preprocessing Ablation: Pipeline A vs Pipeline B",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, 110)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================= ERROR ANALYSIS ==============================

def generate_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: pd.DataFrame,
    label_encoder,
    languages: list,
    n_per_lang: int = 3,
) -> pd.DataFrame:
    """
    Sample misclassified examples for qualitative error analysis.

    Returns a DataFrame with columns:
    [language, text, true_label, predicted_label]
    """
    errors_df = test_df.copy()
    errors_df["predicted"] = label_encoder.inverse_transform(y_pred)
    errors_df["true_label"] = label_encoder.inverse_transform(y_true)
    errors_df["correct"] = errors_df["predicted"] == errors_df["true_label"]

    misclassified = errors_df[~errors_df["correct"]]

    samples = []
    for lang in languages:
        lang_errors = misclassified[misclassified["language"] == lang]
        n = min(n_per_lang, len(lang_errors))
        if n > 0:
            sampled = lang_errors.sample(n=n, random_state=42)
            samples.append(sampled)

    if not samples:
        return pd.DataFrame(columns=["language", "text", "true_label", "predicted"])

    result = pd.concat(samples, ignore_index=True)
    result["language_name"] = result["language"].map(LANG_DISPLAY)
    return result[["language_name", "text", "true_label", "predicted"]].rename(
        columns={"language_name": "Language", "true_label": "True Label",
                 "predicted": "Predicted", "text": "Text"}
    )
