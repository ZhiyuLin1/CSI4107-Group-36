# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Loads stopwords from a text file, where each line contains one stopword.
# Returns a set of stopwords.
def load_stopwords(filepath="stopwords.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        # Read lines, strip any whitespace, and ignore blank lines.
        stopwords = {line.strip() for line in f if line.strip()}
    return stopwords

STOP_WORDS = load_stopwords("stopwords.txt")
stemmer = PorterStemmer()


# Converts text to lowercase and splits it into tokens using NLTK's word_tokenize.
def tokenize(text):
    return word_tokenize(text.lower())


# Filters out tokens that are not alphabetic
def remove_non_alphabetic(tokens):
    filtered_tokens = []
    for token in tokens:
        if token.isalpha():
            filtered_tokens.append(token)
    return filtered_tokens



# Removes tokens that appear in the provided stop_words set.
def remove_stopwords(tokens, stop_words = STOP_WORDS):
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens


# Applies the Porter stemmer to each token.
def stem_tokens(tokens):
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens



def preprocess_text(text, apply_stemming=True):
    """
    Preprocesses input text by:
      1. Tokenizing.
      2. Removing non-alphabetic tokens.
      3. Removing stopwords.
      4. Optionally applying stemming.
    Returns a list of cleaned tokens.
    """
    tokens = tokenize(text)
    tokens = remove_non_alphabetic(tokens)
    tokens = remove_stopwords(tokens)

    if apply_stemming:
        tokens = stem_tokens(tokens)
    return tokens

'''
if __name__ == "__main__":
    sample_text = "This is a sample document! It includes numbers like 123 and punctuation."
    processed_tokens = preprocess_text(sample_text)
    print("Processed tokens:", processed_tokens)
'''
