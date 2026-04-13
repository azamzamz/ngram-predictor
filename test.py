from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from dotenv import load_dotenv
import os

load_dotenv("config/.env", override=True)

model = NGramModel(4, 3)
model.load(os.getenv("MODEL"), os.getenv("VOCAB"))

# Check 3gram context
context = ["looked", "at"]
candidates = model.lookup(context)
sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
print("Top 10 predictions for 'looked at':")
for word, prob in sorted_candidates[:10]:
    print(f"  {word}: {prob:.4f}")