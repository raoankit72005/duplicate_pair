# Duplicate Question Pairs

A beginner-friendly ML project that predicts whether two questions ask the same thing. It uses simple Python functions, Bag of Words, 22 comparison features, Random Forest, optional XGBoost, and a small Streamlit app.

| Question 1 | Question 2 | Intended label |
| --- | --- | --- |
| How can I learn Python? | How do I start learning Python? | Duplicate (1) |
| How can I gain weight? | How can I lose weight? | Different (0) |

These examples explain the task, not guaranteed model predictions. The project compares questions; it does not answer them.

## Quick start

Use Python 3.11 or 3.12. Run all commands from the repository folder.

```bash
git clone https://github.com/raoankit72005/duplicate_pair.git
cd duplicate_pair
python -m venv .venv
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Or Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install -r requirements.txt
python download_data.py
python train.py
python -m streamlit run app.py
```

**Train before using the app.** No trained model or dataset is bundled. Training creates `artifacts/model.pkl`.

The default uses 30,000 pairs, up to 3,000 vocabulary words, and 100 trees per model. Runtime depends on your CPU. Sparse matrices avoid creating large dense text matrices.

For a faster first run:

```bash
python train.py --sample-size 3000 --max-features 1000 --trees 30
```

Use all cleaned pairs (slower):

```bash
python train.py --sample-size 0
```

Also compare XGBoost:

```bash
python -m pip install -r requirements-optional.txt
python train.py --with-xgboost
```

Predict from the terminal after training:

```bash
python predict.py "How can I learn Python?" "How do I start learning Python?"
```

## Dataset

The downloader uses the labelled **training split** of [GLUE QQP on Hugging Face](https://huggingface.co/datasets/nyu-mll/glue/viewer/qqp/train). It renames `label` to `is_duplicate` and saves `data/train.csv`.

This is the GLUE version of Quora Question Pairs, not the complete original Kaggle training file. The script makes its own train/validation/test split from those labelled rows. It does not use GLUE's hidden-label test set or claim an official GLUE score.

Alternatively, download the labelled `train.csv` from the [Quora Question Pairs competition](https://www.kaggle.com/competitions/quora-question-pairs/data) and put it in `data/`. Skip the downloader, or supply a different path:

```bash
python train.py --data path/to/train.csv
```

Required columns: `question1`, `question2`, `is_duplicate`. Labels must be 0 or 1. IDs are not model inputs. Review the dataset provider's terms before use or redistribution.

The automatic download needs internet access. If it fails, use the manual CSV route above. Real-data download and benchmark training were not completed in the development environment; the included synthetic smoke test verifies program operation, not prediction quality.

## Learn the project phase by phase

### 1. Define the problem

This is supervised binary classification. One example contains two questions and a duplicate label. Same topic does not imply same intent: learning Python and installing Python are different requests.

### 2. Clean and explore

`prepare_data` removes missing/empty questions, validates labels, and removes repeated pairs, including reversed pairs. All copies of a cleaned pair with conflicting labels are excluded.

Cleaning lowercases, decodes HTML entities, removes tags, expands several contractions, and collapses spaces. It preserves numbers, negation and symbols useful for C++. It is intentionally simple, not a complete language normalizer.

Training produces label and question-length plots plus a data summary.

### 3. Split before fitting

Use a stratified 70% training, 15% validation, 15% test split, with seed 42 by default. The vocabulary is fitted only on training questions. Validation chooses the model and threshold. The selected model is evaluated on test data afterwards.

**Limitation:** the same individual question can occur in different splits with different partners. Exact cleaned pairs cannot. Results therefore measure unseen pairs, not necessarily entirely unseen questions. The data summary reports overlapping question counts. A stricter experiment would place connected groups of questions in a single split.

### 4. Build Bag of Words

`CountVectorizer` learns a shared vocabulary from both training question columns. Each question becomes a vector of word counts; the pair is the concatenation of two vectors.

With 3,000 vocabulary words, a pair has 6,000 text features. The actual count can be smaller if the vocabulary is smaller.

Before extracting features, each cleaned pair is sorted alphabetically. This makes swapping the two input questions produce exactly the same features and prediction. This is a deliberate change from arbitrary question ordering.

### 5. Add seven basic features

| Features | Meaning |
| --- | --- |
| q1_characters, q2_characters | Character counts |
| q1_words, q2_words | Word counts including repetition |
| common_words | Shared unique-word count |
| total_unique_words | Sum of unique-word counts in both questions |
| word_share | Shared count divided by that sum |

For `how to learn python` and `how to study python`, three words are shared and the total is eight. Word share is 3/8 = 0.375. Its maximum is 0.5; it is not Jaccard similarity.

### 6. Add fifteen advanced features

**Eight token features:** minimum- and maximum-normalized overlap for content words, stopwords and all words (six features), plus first-token and last-token matches (two). Stopwords use scikit-learn's list; they are still retained in Bag of Words.

**Three length features:** mean word count, absolute word-count difference, and longest shared contiguous character substring length divided by shorter character length.

**Four fuzzy features:** RapidFuzz ratio, partial ratio, token-sort ratio and token-set ratio, on a 0–100 scale. These measure string similarity, not full meaning. RapidFuzz replaces the tutorial's older FuzzyWuzzy dependency, so exact scores may differ.

Empty denominators use `max(..., 1)`. `FEATURE_NAMES` lists all 22 engineered features in order. With 3,000 vocabulary words, the full representation has 6,022 columns.

### 7. Train and compare

| Experiment | Input |
| --- | --- |
| rf_bow | Bag of Words |
| rf_basic | Bag of Words + seven basic features |
| rf_full | Bag of Words + 22 engineered features |
| xgb_full (optional) | Bag of Words + 22 engineered features |

Random Forest combines decision trees. XGBoost adds trees sequentially to improve predictions. Application code uses small functions and direct estimator calls, without a complex pipeline or custom model classes.

For each candidate, try thresholds from 0.30 to 0.80 in steps of 0.05 and choose the best validation F1. Then select the candidate with the highest validation F1. Ties keep the earlier threshold/model. The saved model remains fitted on training data; test data is never used to train it.

F1 is an educational choice. Automatic question merging should instead prioritize a suitable precision requirement on validation data. Raw classifier scores are not calibrated probabilities.

### 8. Evaluate and inspect errors

Reports contain accuracy, duplicate-class precision, recall, F1, ROC-AUC, log loss, and a confusion matrix. A majority-class baseline is included.

A false positive incorrectly marks different questions as duplicates. A false negative misses a real duplicate. Inspect `test_errors.csv` to understand failures, but repeated tuning on these errors makes the test set development data; reserve a new final test set for subsequent experiments.

No accuracy is hardcoded or promised. Use your generated results rather than the video's numbers.

### 9. Save and predict

The selected model, vocabulary, feature mode, threshold and seed are saved together in `artifacts/model.pkl`. Prediction uses the same feature functions as training. Load only pickle files you generated or trust.

### 10. Use the app

Enter two questions and click **Compare questions**. Empty inputs are rejected and each question is limited to 2,000 characters. The app displays a predicted class and duplicate score. It caches the model and reloads it when the saved file changes.

## Files

| File | Purpose |
| --- | --- |
| download_data.py | Download labelled GLUE QQP data |
| features.py | Clean, order, vectorize and compare questions |
| train.py | Split, train, compare, evaluate and save |
| predict.py | Shared prediction function and terminal command |
| app.py | Streamlit interface |
| tests/test_project.py | Feature, cleaning, split and input checks |
| tests/test_smoke.py | Tiny synthetic training and application check |
| .github/workflows/checks.yml | Run checks on GitHub |

Generated reports: `data_summary.json`, `split_assignments.csv`, `eda.png`, `validation_results.csv`, `test_metrics.json`, `classification_report.txt`, `confusion_matrix.png`, and `test_errors.csv`.

Dataset files, models, and reports are ignored by Git.

Use `--output experiments/run1` for a separate experiment. The app loads the default `artifacts/model.pkl`, so use the default output directory for the model you want to serve.

## Colab

Run in a new notebook:

```python
!git clone https://github.com/raoankit72005/duplicate_pair.git
%cd duplicate_pair
!pip install -r requirements.txt
!python download_data.py
!python train.py --sample-size 30000
```

Then:

```python
from predict import load_model, predict_pair

artifacts = load_model()
label, score = predict_pair(
    "How can I learn Python?",
    "How do I study Python?",
    artifacts
)
print("Duplicate" if label else "Not duplicate", score)
```

Download the model and reports before the session ends. Colab does not automatically expose the Streamlit port as a public website.

## Deployment

On a Python server, install dependencies and train once, or transfer your generated model from your training machine. Then run:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Use the same dependency versions during training and serving. Save them using `python -m pip freeze > requirements-trained.txt`. Do not train on each web request.

Deploying source alone displays a missing-model message; your host also needs `artifacts/model.pkl`. Public hosting, HTTPS and access settings depend on your provider and are not configured here.

## Tests

```bash
python -m unittest discover -s tests -v
```

The smoke test uses deliberately simple synthetic examples, a temporary directory and a few trees. Its accuracy is **not** a Quora benchmark. It checks training, saved-model prediction, question swapping, app loading and empty-input handling. If XGBoost is installed, it checks that candidate too. No real dataset download is required for tests.

## Limitations and extensions

Word counts and fuzzy matching can miss synonyms, negation, named entities and changes in intent. This project handles English question pairs; it does not search a question database. Try TF-IDF, bigrams, more data, stricter splits, or sentence embeddings as measured extensions.

Inspired by [CampusX's Duplicate Question Pairs tutorial](https://www.youtube.com/watch?v=1fvQU5yPjFs). This implementation was written for this repository, with train-only vocabulary fitting, pair deduplication before splitting, consistent question ordering, validation-based selection, and held-out test evaluation.
