# N-Gram Next-Word Predictor

A next-word prediction system built from scratch using an n-gram language model trained on four Sherlock Holmes novels by Arthur Conan Doyle.

## Requirements
- Python 3.11
- Install dependencies with: `pip install -r requirements.txt`

## Setup
1. Clone the repository: `git clone https://github.com/yourusername/ngram-predictor.git`
2. Create and activate the Anaconda environment: `conda create -n ngram-predictor python=3.11` then `conda activate ngram-predictor`
3. Install dependencies: `pip install -r requirements.txt`
4. Fill in `config/.env` with the required variables (see `.env` section below)
5. Download the four training books from Project Gutenberg and place them in `data/raw/train/`

## Usage
python main.py --step dataprep
python main.py --step model
python main.py --step inference
python main.py --step all

## Project Structure
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── eval/
│   ├── processed/
│   └── model/
├── src/
│   ├── data_prep/
│   │   └── normalizer.py
│   ├── model/
│   │   └── ngram_model.py
│   ├── inference/
│   │   └── predictor.py
│   └── evaluation/
│       └── evaluator.py
├── main.py
├── requirements.txt
└── .gitignore