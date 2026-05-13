### H2 Contribution Statements

**Version 1 — Negative result (honest finding):**

Contrary to initial expectations, the performance gap between TF-IDF and transformer models was not consistently larger for non-Latin-script languages; character n-gram TF-IDF with Complement Naive Bayes achieved competitive or superior accuracy to mBERT on Tamil and Swahili, demonstrating that the advantage of shared semantic representations over well-configured classical baselines is neither uniform nor predictable from script type alone.

**Version 2 — Positive discovery (reframed):**

An unexpected finding revealed that character n-gram TF-IDF paired with Complement Naive Bayes achieves competitive performance with mBERT on morphologically rich languages such as Tamil and Swahili, suggesting that subword-level statistical features can partially compensate for the absence of shared semantic representations when the target language exhibits productive agglutinative morphology.
