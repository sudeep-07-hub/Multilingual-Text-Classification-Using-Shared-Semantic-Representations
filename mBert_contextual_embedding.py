from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from cleaning_pipeline_B import train_df_combined, test_df_combined

# Load the mBERT model and tokenizer
mbert_tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
mbert_model = AutoModel.from_pretrained('bert-base-multilingual-cased')
mbert_model.eval()

def get_mbert_embedding(text, batch_size=16):

    all_embeddings = []
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        encode = mbert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
    
        with torch.no_grad():
            outputs = mbert_model(**encode)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)

    return np.vstack(all_embeddings)

X_train_mbert = get_mbert_embedding(
    train_df_combined['text_pipeline_b'].tolist()
)

X_test_mbert = get_mbert_embedding(
    test_df_combined['text_pipeline_b'].tolist()
)