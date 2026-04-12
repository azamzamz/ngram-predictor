"""
Main entry point for the N-Gram Next-Word Predictor.
Loads configuration, instantiates all objects, and runs the selected pipeline step.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv("config/.env", override=True)

sys.path.append(os.path.dirname(__file__))

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


def run_dataprep(normalizer, train_dir, train_tokens):
    """
    Run the data preparation pipeline.

    Parameters:
        normalizer (Normalizer): A Normalizer instance.
        train_dir (str): Path to the raw training data folder.
        train_tokens (str): Path to save the tokenized output file.

    Returns:
        None
    """
    print("Loading raw text...")
    text = normalizer.load(train_dir)

    print("Stripping Gutenberg header and footer...")
    text = normalizer.strip_gutenberg(text)

    print("Sentence tokenizing...")
    sentences = normalizer.sentence_tokenize(text)

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


def run_model(model, train_tokens, model_path, vocab_path):
    """
    Run the model training pipeline.

    Parameters:
        model (NGramModel): An NGramModel instance.
        train_tokens (str): Path to the tokenized training file.
        model_path (str): Path to save model.json.
        vocab_path (str): Path to save vocab.json.

    Returns:
        None
    """
    print("Building vocabulary...")
    model.build_vocab(train_tokens)

    print("Building counts and probabilities...")
    model.build_counts_and_probabilities(train_tokens)

    print("Saving model...")
    model.save_model(model_path)
    model.save_vocab(vocab_path)
    print("Model training complete!")


def run_inference(predictor):
    """
    Run the interactive CLI prediction loop.

    Parameters:
        predictor (Predictor): A Predictor instance.

    Returns:
        None
    """
    print("\nN-Gram Next-Word Predictor")
    print("Type a sequence of words to get predictions.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() == "quit":
                print("Goodbye.")
                break
            if not user_input:
                print("Please type at least one word.")
                continue
            predictions = predictor.predict_next(user_input)
            print(f"Predictions: {predictions}\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


def main():
    """
    Main function — parses arguments, loads config, instantiates objects,
    and runs the selected pipeline step.
    """
    parser = argparse.ArgumentParser(description="N-Gram Next-Word Predictor")
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        required=True,
        help="Pipeline step to run: dataprep, model, inference, or all"
    )
    args = parser.parse_args()

    # Load config
    train_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    ngram_order = int(os.getenv("NGRAM_ORDER"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))
    top_k = int(os.getenv("TOP_K"))

    # Instantiate objects once
    normalizer = Normalizer()
    model = NGramModel(ngram_order, unk_threshold)

    if args.step == "dataprep":
        run_dataprep(normalizer, train_dir, train_tokens)

    elif args.step == "model":
        run_model(model, train_tokens, model_path, vocab_path)

    elif args.step == "inference":
        model.load(model_path, vocab_path)
        predictor = Predictor(model, normalizer, ngram_order, top_k)
        run_inference(predictor)

    elif args.step == "all":
        run_dataprep(normalizer, train_dir, train_tokens)
        run_model(model, train_tokens, model_path, vocab_path)
        predictor = Predictor(model, normalizer, ngram_order, top_k)
        run_inference(predictor)


if __name__ == "__main__":
    main()