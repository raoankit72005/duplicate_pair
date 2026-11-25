import argparse
import pickle
from pathlib import Path

import pandas as pd

from features import clean_text, make_matrix


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.pkl"


def load_model(path=MODEL_PATH):
    with open(path, "rb") as file:
        return pickle.load(file)


def predict_pair(q1, q2, artifacts):
    if not clean_text(q1) or not clean_text(q2):
        raise ValueError("Please enter two non-empty questions.")
    if max(len(q1), len(q2)) > 2000:
        raise ValueError("Please keep each question within 2,000 characters.")
    data = pd.DataFrame({"question1": [q1], "question2": [q2]})
    matrix = make_matrix(data, artifacts["vectorizer"], artifacts["mode"])
    score = float(artifacts["model"].predict_proba(matrix)[0, 1])
    return int(score >= artifacts["threshold"]), score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question1")
    parser.add_argument("question2")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run python train.py first.")
    label, score = predict_pair(args.question1, args.question2, load_model())
    print("Duplicate" if label else "Not duplicate")
    print(f"Duplicate score: {score:.3f}")


if __name__ == "__main__":
    main()
