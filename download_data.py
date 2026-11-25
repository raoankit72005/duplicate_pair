import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/train.csv")
    args = parser.parse_args()
    dataset = load_dataset("nyu-mll/glue", "qqp", split="train")
    data = dataset.to_pandas().rename(columns={"label": "is_duplicate"})
    data = data[["question1", "question2", "is_duplicate"]]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Saved {len(data):,} labelled GLUE QQP training pairs to {path}")
    print("This is the GLUE training subset, not the complete Kaggle training file.")


if __name__ == "__main__":
    main()
