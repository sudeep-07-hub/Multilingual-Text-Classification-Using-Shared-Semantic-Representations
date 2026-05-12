import re
from cleaning_pipeline_A import train_df_combined, test_df_combined


def preprocessing_b(text):

    text = text.strip()
    text = re.sub(r'\s+', '', text)
    return text


train_df_combined['text_pipeline_b'] = train_df_combined['text'].apply(
    preprocessing_b
)

test_df_combined['text_pipeline_b'] = test_df_combined['text'].apply(
    preprocessing_b
)
