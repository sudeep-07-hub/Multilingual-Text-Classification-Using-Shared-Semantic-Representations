from dataset_loader import all_data, languages
import pandas as pd

SAMPLE_PER_LANG_TRAIN = 400
SAMPLE_PER_LANG_TEST = 100

sampled_train_frame = []
sampled_test_frame = []

for lang in languages:

    train_df = all_data[lang]['train'].sample(
        min(SAMPLE_PER_LANG_TRAIN, len(all_data[lang]['train'])),
        random_state=42
    )
    test_df = all_data[lang]['test'].sample(
        min(SAMPLE_PER_LANG_TEST, len(all_data[lang]['test'])),
        random_state=42
    )
    train_df['language'] = lang
    test_df['language'] = lang
    sampled_train_frame.append(train_df)
    sampled_test_frame.append(test_df)

train_df_combined = pd.concat(sampled_train_frame, ignore_index=True)
test_df_combined = pd.concat(sampled_test_frame, ignore_index=True)
