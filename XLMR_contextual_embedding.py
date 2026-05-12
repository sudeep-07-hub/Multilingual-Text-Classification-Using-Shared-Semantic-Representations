from transformers import AutoModel, AutoTokenizer
from mBert_contextual_embedding import get_mbert_embedding
import torch
from cleaning_pipeline_B import train_df_combined, test_df_combined

xlmr_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
xlmr_model = AutoModel.from_pretrained('xlm-roberta-base')
xlmr_model.eval()

X_train_xlmr = get_mbert_embedding.__wrapped__(
    train_df_combined['text_pipeline_b'].tolist(),
    tokenizer=xlmr_tokenizer,
    model=xlmr_model
)

X_test_xlmr = get_mbert_embedding.__wrapped__(
    test_df_combined['text_pipeline_b'].tolist(),
    tokenizer=xlmr_tokenizer,
    model=xlmr_model
)