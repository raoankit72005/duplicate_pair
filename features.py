import html
import re
from difflib import SequenceMatcher

import numpy as np
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


FEATURE_NAMES = [
    "q1_characters", "q2_characters", "q1_words", "q2_words",
    "common_words", "total_unique_words", "word_share",
    "common_content_min", "common_content_max",
    "common_stop_min", "common_stop_max",
    "common_tokens_min", "common_tokens_max",
    "first_word_equal", "last_word_equal", "mean_word_count",
    "word_count_difference", "longest_substring_ratio",
    "fuzzy_ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"
]


def clean_text(text):
    text = html.unescape(str(text)).lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    for short, long in {"can't": "cannot", "won't": "will not", "n't": " not",
                        "'re": " are", "'ve": " have", "'ll": " will"}.items():
        text = text.replace(short, long)
    text = re.sub(r"[^\w\s+#.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_pair(q1, q2):
    return tuple(sorted([clean_text(q1), clean_text(q2)]))


def overlap_ratios(words1, words2):
    common = len(words1 & words2)
    return [common / max(min(len(words1), len(words2)), 1),
            common / max(max(len(words1), len(words2)), 1)]


def pair_features(q1, q2, mode="full"):
    tokens1, tokens2 = q1.split(), q2.split()
    words1, words2 = set(tokens1), set(tokens2)
    common = len(words1 & words2)
    total = len(words1) + len(words2)
    values = [len(q1), len(q2), len(tokens1), len(tokens2),
              common, total, common / max(total, 1)]
    if mode == "basic":
        return values
    stop1, stop2 = words1 & ENGLISH_STOP_WORDS, words2 & ENGLISH_STOP_WORDS
    values += overlap_ratios(words1 - stop1, words2 - stop2)
    values += overlap_ratios(stop1, stop2)
    values += overlap_ratios(words1, words2)
    first_equal = bool(tokens1 and tokens2 and tokens1[0] == tokens2[0])
    last_equal = bool(tokens1 and tokens2 and tokens1[-1] == tokens2[-1])
    longest = SequenceMatcher(None, q1, q2, autojunk=False).find_longest_match().size
    values += [int(first_equal), int(last_equal), (len(tokens1) + len(tokens2)) / 2,
               abs(len(tokens1) - len(tokens2)), longest / max(min(len(q1), len(q2)), 1)]
    values += [fuzz.ratio(q1, q2), fuzz.partial_ratio(q1, q2),
               fuzz.token_sort_ratio(q1, q2), fuzz.token_set_ratio(q1, q2)]
    return values


def make_matrix(data, vectorizer, mode="full"):
    if mode not in {"bow", "basic", "full"}:
        raise ValueError("mode must be bow, basic, or full")
    pairs = [canonical_pair(a, b) for a, b in zip(data["question1"], data["question2"])]
    q1, q2 = zip(*pairs)
    blocks = [vectorizer.transform(q1), vectorizer.transform(q2)]
    if mode != "bow":
        extra = np.array([pair_features(a, b, mode) for a, b in pairs], dtype=np.float32)
        blocks.append(csr_matrix(extra))
    return hstack(blocks, format="csr", dtype=np.float32)
