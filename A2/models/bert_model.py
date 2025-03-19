"""
BERT Model for generating sentence embeddings using SentenceTransformer.
This module encapsulates loading the model and generating embeddings for given texts.
"""

from sentence_transformers import SentenceTransformer


class BertEmbedder:
    def __init__(self, model_name="sentence-transformers/bert-base-nli-mean-tokens"):
        """
        Initializes the BERT embedder by loading a pre-trained model.
        :param model_name: Name or path of the pre-trained SentenceTransformer model.
        """
        print("Loading BERT model...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encodes one or more texts into embeddings.

        :param texts: A single string or a list of strings.
        :return: For a single text, returns a 1D numpy array embedding.
                 For a list of texts, returns a list of numpy array embeddings.
        """
        if isinstance(texts, list):
            embeddings = self.model.encode(texts)
        else:
            embeddings = self.model.encode([texts])[0]
        return embeddings


# Example usage (if you want to test this module independently):
if __name__ == "__main__":
    embedder = BertEmbedder()
    sample_text = "This is a sample sentence for embedding."
    embedding = embedder.encode(sample_text)
    print("Embedding shape:", embedding.shape)
