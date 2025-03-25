# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # Suppress TensorFlow logging.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # Disable oneDNN optimizations.

import tensorflow as tf
import tensorflow_hub as hub

class USEEmbedder:
    # Initializes the USE embedder by loading the pre-trained model from TensorFlow Hub.
    def __init__(self, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        print("Loading Universal Sentence Encoder model...")
        self.model = hub.load(model_url)

    # Encodes one or more texts into embeddings using USE.
    def encode(self, texts):
        # Ensure texts is in list format.
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = self.model(texts)
        return embeddings.numpy()

# Example usage (if you want to test this module independently):
if __name__ == "__main__":
    embedder = USEEmbedder()
    sample_text = "This is a sample sentence for USE embedding."
    embedding = embedder.encode(sample_text)
    print("Embedding shape:", embedding.shape)
