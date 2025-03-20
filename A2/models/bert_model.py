# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

from sentence_transformers import SentenceTransformer

class BertEmbedder:

    # Initializes the BERT embedder by loading a pre-trained model.
    def __init__(self, model_name="sentence-transformers/bert-base-nli-mean-tokens"):
        print("Loading BERT model...")
        self.model = SentenceTransformer(model_name)

    # Encodes one or more texts into embeddings.
    def encode(self, texts):
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
