# Hengjing Zhang 300288003
# Tom Cui 300345709
# Zhiyu Lin 300255509

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

