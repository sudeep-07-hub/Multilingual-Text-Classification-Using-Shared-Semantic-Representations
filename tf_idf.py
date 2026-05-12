from sklearn.feature_extraction.text import TfidfVectorizer
from cleaning_pipeline_B import train_df_combined, test_df_combined

tf_idf_vectorizer = TfidfVectorizer(
    max_features=10000,
    analyzer='char_wb',
    ngram_range=(2, 4)
)

X_train_tf_idf = tf_idf_vectorizer.fit_transform(
    train_df_combined['text_pipeline_a']
)

X_test_tf_idf = tf_idf_vectorizer.transform(
    test_df_combined['text_pipeline_a']
)
