"""
Module responsibility: Accepting a pre-loaded NGramModel and Normalizer via the
constructor, normalizing input text, and returning the top-k predicted next words
sorted by probability. Backoff lookup is delegated to NGramModel.lookup().
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


class Predictor:
    """
    Accepts a pre-loaded NGramModel and Normalizer, normalizes input text,
    and returns the top-k predicted next words sorted by probability.
    """

    def __init__(self, model, normalizer, ngram_order, top_k):
        """
        Accept a pre-loaded NGramModel and Normalizer instance.

        Parameters:
            model (NGramModel): A pre-loaded NGramModel instance.
            normalizer (Normalizer): A pre-loaded Normalizer instance.
            ngram_order (int): The n-gram order to use for context extraction.
            top_k (int): Number of top predictions to return.

        Returns:
            None
        """
        self.model = model
        self.normalizer = normalizer
        self.ngram_order = ngram_order
        self.top_k = top_k

    def normalize(self, text):
        """
        Normalize input text and extract the last ngram_order - 1 words as context.

        Parameters:
            text (str): Raw input string from the user.

        Returns:
            list: List of context words (last ngram_order - 1 words).
        """
        normalized = self.normalizer.normalize(text)
        words = normalized.split()
        context = words[-(self.ngram_order - 1):]
        return context

    def map_oov(self, context):
        """
        Replace out-of-vocabulary words with <UNK>.

        Parameters:
            context (list): List of context words.

        Returns:
            list: Context with OOV words replaced by <UNK>.
        """
        return [w if w in self.model.vocab else "<UNK>" for w in context]

    def predict_next(self, text, k=None):
        """
        Orchestrate normalize -> map_oov -> NGramModel.lookup() and return
        top-k words sorted by probability.

        Parameters:
            text (str): Raw input string from the user.
            k (int): Number of predictions to return. Defaults to self.top_k.

        Returns:
            list: Top-k predicted next words sorted by probability.
        """
        if k is None:
            k = self.top_k

        context = self.normalize(text)
        context = self.map_oov(context)
        candidates = self.model.lookup(context)

        if not candidates:
            return []

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_candidates[:k]]


def main():
    """
    Entry point for running the Predictor module in isolation.
    Tests prediction on a sample input.
    """
    from dotenv import load_dotenv
    load_dotenv("config/.env", override=True)

    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    top_k = int(os.getenv("TOP_K"))

    normalizer = Normalizer()
    model = NGramModel(ngram_order, unk_threshold=3)
    model.load(model_path, vocab_path)

    predictor = Predictor(model, normalizer, ngram_order, top_k)

    test_inputs = [
        "holmes looked at",
        "the game is",
        "zzz qqq"
    ]

    for text in test_inputs:
        predictions = predictor.predict_next(text)
        print(f"> {text}")
        print(f"Predictions: {predictions}")
        print()


if __name__ == "__main__":
    main()