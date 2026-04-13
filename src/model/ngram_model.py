"""
Module responsibility: Building, storing, and exposing n-gram probability tables
and backoff lookup across all orders from 1 up to NGRAM_ORDER.
"""

import os
import json
from collections import defaultdict


class NGramModel:
    """
    Builds and stores n-gram probability tables at all orders from 1 up to NGRAM_ORDER.
    Provides backoff lookup to return the most probable next words given a context.
    """

    def __init__(self, ngram_order, unk_threshold):
        """
        Initialize the NGramModel.

        Parameters:
            ngram_order (int): Maximum n-gram order to build.
            unk_threshold (int): Minimum frequency for a word to stay in vocabulary.
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.probabilities = {}

    def build_vocab(self, token_file):
        """
        Build vocabulary from token file. Replace words appearing fewer than
        unk_threshold times with <UNK>.

        Parameters:
            token_file (str): Path to the tokenized training file.

        Returns:
            None
        """
        word_counts = defaultdict(int)

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    word_counts[word] += 1

        self.vocab = set()
        for word, count in word_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(word)

        self.vocab.add("<UNK>")
        print(f"Vocabulary built: {len(self.vocab)} words")

    def _replace_unk(self, words):
        """
        Replace words not in vocabulary with <UNK>.

        Parameters:
            words (list): List of word strings.

        Returns:
            list: List with OOV words replaced by <UNK>.
        """
        return [w if w in self.vocab else "<UNK>" for w in words]

    def build_counts_and_probabilities(self, token_file):
        """
        Count all n-grams at orders 1 through ngram_order and compute MLE
        probabilities for each order.

        Parameters:
            token_file (str): Path to the tokenized training file.

        Returns:
            None
        """
        counts = {order: defaultdict(int) for order in range(1, self.ngram_order + 1)}

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                words = self._replace_unk(line.strip().split())
                for order in range(1, self.ngram_order + 1):
                    for i in range(len(words) - order + 1):
                        ngram = tuple(words[i:i + order])
                        counts[order][ngram] += 1

        self.probabilities = {}

        # Unigram probabilities
        total_words = sum(counts[1].values())
        self.probabilities["1gram"] = {}
        for (word,), count in counts[1].items():
            self.probabilities["1gram"][word] = count / total_words

        # Higher order probabilities
        for order in range(2, self.ngram_order + 1):
            key = f"{order}gram"
            self.probabilities[key] = {}
            for ngram, count in counts[order].items():
                context = ngram[:-1]
                word = ngram[-1]
                context_count = counts[order - 1][context]
                if context_count > 0:
                    context_str = " ".join(context)
                    if context_str not in self.probabilities[key]:
                        self.probabilities[key][context_str] = {}
                    self.probabilities[key][context_str][word] = count / context_count

        print(f"Probabilities built for orders 1 to {self.ngram_order}")

    def lookup(self, context):
        """
        Backoff lookup: try the highest-order context first, fall back to lower
        orders down to 1-gram. Return a dict of word probabilities from the
        highest order that matches and has more than one candidate.

        Parameters:
            context (list): List of context words.

        Returns:
            dict: Dictionary of {word: probability} from the best matching order.
                  Returns empty dict if no match at any order.
        """
        context = self._replace_unk(context)

        for order in range(self.ngram_order, 0, -1):
            if order == 1:
                return self.probabilities.get("1gram", {})
            ctx = context[-(order - 1):]
            ctx_str = " ".join(ctx)
            key = f"{order}gram"
            if key in self.probabilities and ctx_str in self.probabilities[key]:
                candidates = self.probabilities[key][ctx_str]
                if len(candidates) > 1:
                    return candidates

        return self.probabilities.get("1gram", {})

    def save_model(self, model_path):
        """
        Save all probability tables to model.json.

        Parameters:
            model_path (str): Path to save model.json.

        Returns:
            None
        """
        folder = os.path.dirname(model_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.probabilities, f, indent=2)
        print(f"Model saved to {model_path}")

    def save_vocab(self, vocab_path):
        """
        Save vocabulary list to vocab.json.

        Parameters:
            vocab_path (str): Path to save vocab.json.

        Returns:
            None
        """
        folder = os.path.dirname(vocab_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(list(self.vocab), f, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    def load(self, model_path, vocab_path):
        """
        Load model.json and vocab.json into the instance.

        Parameters:
            model_path (str): Path to model.json.
            vocab_path (str): Path to vocab.json.

        Returns:
            None
        """
        with open(model_path, "r", encoding="utf-8") as f:
            self.probabilities = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = set(json.load(f))
        print(f"Model loaded from {model_path}")


def main():
    """
    Entry point for running the NGramModel module in isolation.
    Builds vocabulary and probabilities from the training token file.
    """
    from dotenv import load_dotenv
    load_dotenv("config/.env", override=True)

    token_file = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))

    model = NGramModel(ngram_order, unk_threshold)

    print("Building vocabulary...")
    model.build_vocab(token_file)

    print("Building counts and probabilities...")
    model.build_counts_and_probabilities(token_file)

    print("Saving model...")
    model.save_model(model_path)
    model.save_vocab(vocab_path)

    print("Done!")


if __name__ == "__main__":
    main()