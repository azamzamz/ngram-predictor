"""
Module responsibility: Loading, cleaning, tokenizing, and saving the corpus.
"""

import os
import re
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize


class Normalizer:
    """
    Responsible for loading raw text files, stripping Gutenberg headers and footers,
    normalizing text, tokenizing into sentences and words, and saving the result.
    """

    def load(self, folder_path):
        """
        Load all .txt files from a folder and return their combined text.

        Parameters:
            folder_path (str): Path to the folder containing .txt files.

        Returns:
            str: Combined text of all .txt files in the folder.
        """
        combined = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    combined += f.read() + "\n"
        return combined

    def strip_gutenberg(self, text):
        """
        Remove Project Gutenberg header and footer from text.

        Parameters:
            text (str): Raw text including Gutenberg header and footer.

        Returns:
            str: Text with header and footer removed.
        """
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

        start_idx = text.find(start_marker)
        if start_idx != -1:
            start_idx = text.find("\n", start_idx) + 1
            text = text[start_idx:]

        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]

        return text

    def lowercase(self, text):
        """
        Convert all text to lowercase.

        Parameters:
            text (str): Input text.

        Returns:
            str: Lowercased text.
        """
        return text.lower()

    def remove_punctuation(self, text):
        """
        Remove all punctuation from text.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text with punctuation removed.
        """
        return re.sub(r'[^\w\s]', '', text)

    def remove_numbers(self, text):
        """
        Remove all numbers from text.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text with numbers removed.
        """
        return re.sub(r'\d+', '', text)

    def remove_whitespace(self, text):
        """
        Remove extra whitespace and blank lines from text.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text with extra whitespace removed.
        """
        lines = text.splitlines()
        cleaned = [" ".join(line.split()) for line in lines if line.strip()]
        return "\n".join(cleaned)

    def normalize(self, text):
        """
        Apply all normalization steps in order: lowercase, remove punctuation,
        remove numbers, remove whitespace.

        Parameters:
            text (str): Input text.

        Returns:
            str: Fully normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        """
        Split text into a list of sentences.

        Parameters:
            text (str): Normalized text.

        Returns:
            list: List of sentence strings.
        """
        return sent_tokenize(text)

    def word_tokenize(self, sentence):
        """
        Split a single sentence into a list of word tokens.

        Parameters:
            sentence (str): A single sentence string.

        Returns:
            list: List of word tokens.
        """
        return word_tokenize(sentence)

    def save(self, sentences, filepath):
        """
        Write tokenized sentences to output file, one sentence per line.

        Parameters:
            sentences (list): List of token lists.
            filepath (str): Path to the output file.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(" ".join(sentence) + "\n")


def main():
    """
    Entry point for running the Normalizer module in isolation.
    Processes the first 100 sentences of the training corpus as a sample.
    """
    from dotenv import load_dotenv
    load_dotenv("config/.env")

    train_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens = os.getenv("TRAIN_TOKENS")

    normalizer = Normalizer()

    print("Loading raw text...")
    text = normalizer.load(train_dir)

    print("Stripping Gutenberg header and footer...")
    text = normalizer.strip_gutenberg(text)

    print("Sentence tokenizing before normalization...")
    sentences = normalizer.sentence_tokenize(text)

    # Only use first 100 sentences during development
    sentences = sentences[:100]

    print("Normalizing and word tokenizing...")
    tokenized = []
    for sentence in sentences:
        normalized = normalizer.normalize(sentence)
        words = normalizer.word_tokenize(normalized)
        if words:
            tokenized.append(words)

    print("Saving tokens...")
    normalizer.save(tokenized, train_tokens)

    print(f"Done! {len(tokenized)} sentences saved to {train_tokens}")


if __name__ == "__main__":
    main()