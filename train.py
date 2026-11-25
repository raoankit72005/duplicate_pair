import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from features import canonical_pair, make_matrix


def prepare_data(data):
    columns = ["question1", "question2", "is_duplicate"]
    if not set(columns).issubset(data.columns):
        raise ValueError(f"CSV must contain these columns: {columns}")
    data = data.dropna(subset=columns)[columns].copy()
    data["is_duplicate"] = pd.to_numeric(data["is_duplicate"], errors="coerce")
    if not data["is_duplicate"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1. Do not use an unlabelled test file.")
    data["is_duplicate"] = data["is_duplicate"].astype(int)
    pairs = [canonical_pair(a, b) for a, b in zip(data.question1, data.question2)]
    data["question1"] = [pair[0] for pair in pairs]
    data["question2"] = [pair[1] for pair in pairs]
    data = data[(data.question1.str.len() > 0) & (data.question2.str.len() > 0)]
    conflicts = data.groupby(["question1", "question2"])["is_duplicate"].transform("nunique")
    data = data[conflicts == 1]
    return data.drop_duplicates(["question1", "question2"]).reset_index(drop=True)


def split_data(data, seed):
    train, rest = train_test_split(data, test_size=0.30, random_state=seed,
                                  stratify=data.is_duplicate)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed,
                                stratify=rest.is_duplicate)
    return train.copy(), val.copy(), test.copy()


def choose_threshold(labels, scores):
    thresholds = np.arange(0.30, 0.81, 0.05)
    results = [f1_score(labels, scores >= value, zero_division=0) for value in thresholds]
    return float(thresholds[np.argmax(results)])


def evaluate(labels, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist()
    }


def questions(data):
    return set(data.question1) | set(data.question2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-xgboost", action="store_true")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    if args.sample_size < 0 or args.max_features < 1 or args.trees < 1:
        parser.error("Use sample-size >= 0, max-features >= 1 and trees >= 1.")
    path = Path(args.data)
    if not path.exists():
        raise SystemExit("Dataset not found. Run python download_data.py or supply --data path/to/train.csv")
    raw = pd.read_csv(path)
    data = prepare_data(raw)
    cleaned_rows = len(data)
    if args.sample_size and args.sample_size < len(data):
        data, _ = train_test_split(data, train_size=args.sample_size,
                                  stratify=data.is_duplicate, random_state=args.seed)
    if len(data) < 40 or data.is_duplicate.nunique() != 2 or data.is_duplicate.value_counts().min() < 10:
        raise SystemExit("Use at least 40 clean pairs, including at least 10 of each class.")
    train, val, test = split_data(data, args.seed)
    output = Path(args.output)
    reports, artifacts = output / "reports", output / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_csv": str(path), "raw_rows": len(raw), "cleaned_rows": cleaned_rows,
        "used_rows": len(data), "seed": args.seed,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "class_counts": data.is_duplicate.value_counts().to_dict(),
        "validation_questions_seen_in_train": len(questions(val) & questions(train)),
        "test_questions_seen_in_train": len(questions(test) & questions(train)),
        "split_note": "Stratified pair split. Individual questions can occur in multiple splits."
    }
    (reports / "data_summary.json").write_text(json.dumps(summary, indent=2))
    split_rows = pd.concat([part.assign(split=name) for part, name in
                            [(train, "train"), (val, "validation"), (test, "test")]])
    split_rows.to_csv(reports / "split_assignments.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    train.is_duplicate.value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set(title="Training labels: 0=different, 1=duplicate", ylabel="Pairs")
    train.question1.str.split().str.len().hist(bins=30, ax=axes[1])
    axes[1].set(title="Training question lengths", xlabel="Words")
    fig.tight_layout()
    fig.savefig(reports / "eda.png")
    plt.close(fig)

    vectorizer = CountVectorizer(max_features=args.max_features,
                                 token_pattern=r"(?u)\b\w[\w+#.]*", dtype=np.float32)
    vectorizer.fit(pd.concat([train.question1, train.question2]))
    print(f"Train: {len(train)}, validation: {len(val)}, test: {len(test)}", flush=True)
    print(f"Vocabulary fitted only on training: {len(vectorizer.vocabulary_)} words", flush=True)

    candidates = [("rf_bow", "bow"), ("rf_basic", "basic"), ("rf_full", "full")]
    if args.with_xgboost:
        from xgboost import XGBClassifier
        candidates.append(("xgb_full", "full"))

    results = []
    best_model, best_result, best_mode = None, None, None
    for name, mode in candidates:
        print(f"Training {name}...", flush=True)
        X_train = make_matrix(train, vectorizer, mode)
        X_val = make_matrix(val, vectorizer, mode)
        if name.startswith("rf"):
            model = RandomForestClassifier(n_estimators=args.trees, max_depth=25,
                                           min_samples_leaf=2, random_state=args.seed, n_jobs=-1)
        else:
            model = XGBClassifier(n_estimators=args.trees, max_depth=5, learning_rate=0.1,
                                  tree_method="hist", eval_metric="logloss",
                                  random_state=args.seed, n_jobs=2)
        model.fit(X_train, train.is_duplicate)
        scores = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(val.is_duplicate, scores)
        result = {"model": name, **evaluate(val.is_duplicate, scores, threshold)}
        results.append(result)
        print(f"Validation F1: {result['f1']:.4f}, accuracy: {result['accuracy']:.4f}", flush=True)
        if best_result is None or result["f1"] > best_result["f1"]:
            best_model, best_result, best_mode = model, result, mode
        del X_train, X_val

    pd.DataFrame(results).drop(columns="confusion_matrix").to_csv(reports / "validation_results.csv", index=False)
    X_test = make_matrix(test, vectorizer, best_mode)
    scores = best_model.predict_proba(X_test)[:, 1]
    threshold = best_result["threshold"]
    test_metrics = evaluate(test.is_duplicate, scores, threshold)
    majority_class = int(train.is_duplicate.mode()[0])
    test_metrics["majority_baseline_accuracy"] = float((test.is_duplicate == majority_class).mean())
    test_metrics["selected_model"] = best_result["model"]
    (reports / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    predicted = (scores >= threshold).astype(int)
    report = classification_report(test.is_duplicate, predicted, zero_division=0)
    (reports / "classification_report.txt").write_text(report)
    errors = test.copy()
    errors["prediction"], errors["duplicate_score"] = predicted, scores
    errors[errors.is_duplicate != errors.prediction].to_csv(reports / "test_errors.csv", index=False)
    ConfusionMatrixDisplay.from_predictions(test.is_duplicate, predicted,
                                            display_labels=["Different", "Duplicate"])
    plt.tight_layout()
    plt.savefig(reports / "confusion_matrix.png")
    plt.close()

    saved = {"model": best_model, "vectorizer": vectorizer, "mode": best_mode,
             "threshold": threshold, "name": best_result["model"], "seed": args.seed}
    with open(artifacts / "model.pkl", "wb") as file:
        pickle.dump(saved, file)
    print(f"Selected using validation F1: {best_result['model']}")
    print(report)
    print(f"Model saved to {artifacts / 'model.pkl'}")
    print(f"Metrics, plots, split assignments and errors saved to {reports}")


if __name__ == "__main__":
    main()
