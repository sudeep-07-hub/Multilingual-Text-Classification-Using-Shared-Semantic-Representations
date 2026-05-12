import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sampling import train_df_combined, test_df_combined

# Download NLTK resources if not already downloaded

# nltk.download('stopwords')
# nltk.download('punkt_tab')

english_stopwords = stopwords.words('english')


def preprocessing_a(text, language='eng_Latn'):

    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', "", text)  # Remove numbers
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    if language == 'eng_Latn':
        # Remove stopwords
        tokens = [t for t in tokens if t not in english_stopwords]
    tokens = [t for t in tokens if len(t) < 1]  # Remove short words
    return " ".join(tokens)


train_df_combined['text_pipeline_a'] = train_df_combined.apply(
    lambda row: preprocessing_a(row['text'], row['language']), axis=1
)

test_df_combined['text_pipeline_a'] = test_df_combined.apply(
    lambda row: preprocessing_a(row['text'], row['language']), axis=1
)
