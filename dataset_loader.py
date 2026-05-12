from datasets import load_dataset
import pandas as pd

# Load SIB-200 for each language
languages = ['eng_Latn', 'hin_Deva', 'kan_Knda', 'swh_Latn']
all_data = {}

for lang in languages:
    dataset = load_dataset('Davlan/sib200', lang)
    all_data[lang] = {
        'train': dataset['train'].to_pandas(),
        'test': dataset['test'].to_pandas()
    }
