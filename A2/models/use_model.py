import tensorflow as tf
import tensorflow_hub as hub


class USEEmbedder:
    def __init__(self, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        """
        Initializes the USE embedder by loading the pre-trained model from TensorFlow Hub.
        :param model_url: URL for the pre-trained USE model.
        """
        print("Loading Universal Sentence Encoder model...")
        self.model = hub.load(model_url)

    def encode(self, texts):
        """
        Encodes one or more texts into embeddings using USE.

        :param texts: A single string or a list of strings.
        :return: For a single text, returns a 1D numpy array embedding.
                 For a list of texts, returns a 2D numpy array with each row as an embedding.
        """
        # Ensure texts is in list format
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