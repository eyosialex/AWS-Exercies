"""
AWS ML - Week 03: Breaking the SMS Spam Filter (Multinomial Naive Bayes & TF-IDF)
================================================================================
This script implements the complete Week 03 pipeline and hands-on task:
1. Ingestion of raw dataset (SMS Spam Collection) with local & remote fallback.
2. Named data cleaning functions:
   - Handling missing values and duplicates
   - Text whitespace and Unicode normalization
   - Type conversions & feature engineering
   - Text length outlier detection using IQR
3. Statistical calculations using NumPy (mean, median, std, percentile, where, unique).
4. Slide 6 Baseline Model:
   - TfidfVectorizer + MultinomialNB (10-line baseline classifier)
   - Accuracy evaluation on train & test sets
5. Hands-on Challenge: "Break the Spam Filter":
   - Testing 5+ adversarial messages (spelling tricks, polite tone, negation, etc.)
   - Prediction logging and in-depth failure mode analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "sms.tsv")
DATA_FALLBACK_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


# ==============================================================================
# 1. Dataset Loading Function
# ==============================================================================
def load_raw_data(filepath: str = DATA_FILE, fallback_url: str = DATA_FALLBACK_URL) -> pd.DataFrame:
    """
    Loads raw SMS spam dataset from the local repository directory.
    If not found locally, falls back to the remote GitHub URL and caches it.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading dataset from local repository: '{filepath}'")
        df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    else:
        print(f"[INFO] Local file not found. Fetching from remote URL: {fallback_url}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.read_csv(fallback_url, sep='\t', header=None, names=['label', 'message'])
        df.to_csv(filepath, sep='\t', header=False, index=False)
        print(f"[INFO] Dataset cached locally at '{filepath}'")

    print(f"[INFO] Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


# ==============================================================================
# 2. Named Data Cleaning Functions
# ==============================================================================
def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and handles missing values and removes duplicate message entries.
    """
    df_clean = df.copy()
    initial_count = len(df_clean)

    # 1. Missing values
    missing_count = df_clean.isna().sum().to_dict()
    print(f"[CLEAN] Missing values per column: {missing_count}")
    df_clean = df_clean.dropna(subset=['label', 'message'])

    # 2. Duplicate records
    duplicate_count = df_clean.duplicated(subset=['message']).sum()
    print(f"[CLEAN] Duplicate messages found: {duplicate_count}")
    df_clean = df_clean.drop_duplicates(subset=['message']).reset_index(drop=True)

    print(f"[CLEAN] Records retained: {len(df_clean)} / {initial_count} (dropped {initial_count - len(df_clean)})")
    return df_clean


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trims leading/trailing whitespace, strips non-printable control characters,
    and standardizes whitespace formatting across text messages.
    """
    df_clean = df.copy()
    # Normalize whitespaces and strip outer padding
    df_clean['message'] = df_clean['message'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df_clean['label'] = df_clean['label'].astype(str).str.strip().str.lower()
    print("[CLEAN] Text whitespace normalization completed.")
    return df_clean


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly converts column data types and creates numeric length features.
    """
    df_clean = df.copy()
    df_clean['message'] = df_clean['message'].astype("string")
    df_clean['label'] = df_clean['label'].astype("category")
    df_clean['char_length'] = df_clean['message'].str.len().astype(np.int64)
    df_clean['word_count'] = df_clean['message'].str.split().str.len().astype(np.int64)
    print(f"[CLEAN] Column data types converted: message -> string, label -> category, length -> int64.")
    return df_clean


def detect_text_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies extreme message character length anomalies using the Interquartile Range (IQR).
    Flags outlier records using NumPy.
    """
    df_clean = df.copy()
    lengths = df_clean['char_length'].to_numpy()

    q25, q75 = np.percentile(lengths, [25, 75])
    iqr = q75 - q25
    lower_bound = max(0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    # Identify outliers using np.where
    df_clean['is_length_outlier'] = np.where(
        (df_clean['char_length'] < lower_bound) | (df_clean['char_length'] > upper_bound),
        True,
        False
    )

    outlier_count = int(np.sum(df_clean['is_length_outlier']))
    print(f"[OUTLIERS] IQR Bounds: [{lower_bound:.1f}, {upper_bound:.1f}] chars. Found {outlier_count} length outliers.")
    return df_clean


# ==============================================================================
# 3. NumPy Calculations
# ==============================================================================
def compute_numpy_statistics(df: pd.DataFrame) -> dict:
    """
    Performs comprehensive statistical calculations using 6 distinct NumPy functions:
    np.mean, np.median, np.std, np.percentile, np.where, np.unique.
    """
    lengths = df['char_length'].to_numpy()
    words = df['word_count'].to_numpy()
    labels = df['label'].to_numpy()

    # 1. np.mean
    mean_len = float(np.mean(lengths))
    mean_words = float(np.mean(words))

    # 2. np.median
    median_len = float(np.median(lengths))
    median_words = float(np.median(words))

    # 3. np.std
    std_len = float(np.std(lengths))
    std_words = float(np.std(words))

    # 4. np.percentile
    p10, p25, p50, p75, p90, p99 = np.percentile(lengths, [10, 25, 50, 75, 90, 99])

    # 5. np.unique
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, [int(c) for c in counts]))

    # 6. np.where for binary classification array
    binary_labels = np.where(labels == 'spam', 1, 0)
    spam_mean_len = float(np.mean(lengths[np.where(binary_labels == 1)]))
    ham_mean_len = float(np.mean(lengths[np.where(binary_labels == 0)]))

    stats = {
        "mean_length": mean_len,
        "median_length": median_len,
        "std_length": std_len,
        "percentiles": {"p10": p10, "p25": p25, "p75": p75, "p90": p90, "p99": p99},
        "mean_words": mean_words,
        "median_words": median_words,
        "std_words": std_words,
        "label_distribution": label_dist,
        "spam_mean_length": spam_mean_len,
        "ham_mean_length": ham_mean_len,
    }

    print("\n" + "="*50)
    print("NUMPY STATISTICAL COMPUTATION SUMMARY")
    print("="*50)
    print(f"Total Messages Analyzed: {len(lengths)}")
    print(f"Class Distribution     : {label_dist}")
    print(f"Character Length Mean  : {mean_len:.2f} chars (std: {std_len:.2f})")
    print(f"Character Length Median: {median_len:.1f} chars (IQR: [{p25:.1f}, {p75:.1f}])")
    print(f"Spam vs Ham Mean Length: Spam = {spam_mean_len:.2f} chars | Ham = {ham_mean_len:.2f} chars")
    print(f"Word Count Mean        : {mean_words:.2f} words (median: {median_words:.1f})")
    print("="*50 + "\n")

    return stats


# ==============================================================================
# 4. Slide 6 Baseline 10-Line Model
# ==============================================================================
def train_slide_6_model(emails, labels):
    """
    Executes the exact 10-line model from Slide 6:
      - train_test_split (test_size=0.2, random_state=42)
      - TfidfVectorizer()
      - MultinomialNB()
      - model.fit(vec.fit_transform(X_train), y_train)
    """
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42)

    vec = TfidfVectorizer()         # text -> numbers (features)
    model = MultinomialNB()         # a simple, fast classifier
    model.fit(vec.fit_transform(X_train), y_train)  # learn

    # Note accuracy
    train_acc = accuracy_score(y_train, model.predict(vec.transform(X_train)))
    test_acc = accuracy_score(y_test, model.predict(vec.transform(X_test)))

    print("\n" + "="*50)
    print("SLIDE 6 BASELINE MODEL ACCURACY")
    print("="*50)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy    : {test_acc * 100:.2f}%")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, model.predict(vec.transform(X_test)), digits=4))
    print("="*50 + "\n")

    return vec, model, X_test, y_test


# ==============================================================================
# 5. Break the Spam Filter: Adversarial Testing & Failure Analysis
# ==============================================================================
def run_adversarial_break_tests(vec: TfidfVectorizer, model: MultinomialNB):
    """
    Executes the Hands-on Challenge:
    Write 5 messages that try to fool it: spam that slips through, or normal mail that gets flagged.
    Records predictions, probabilities, and details why the filter failed.
    """
    test_suite = [
        {
            "id": 1,
            "category": "Adversarial Spam (Spelling / Leetspeak Obfuscation)",
            "message": "Cl@im your fr33 g1ft card now! V1s1t www.r3w@rds-cl@im.com to get your 500 b0nus t0d@y!",
            "true_intent": "spam",
            "trick": "Leetspeak substitution (@ for a, 3 for e, 1 for i, 0 for o) to evade vocabulary lookup.",
            "why_it_fails": (
                "The TfidfVectorizer tokenizes using the default regex r'(?u)\\b\\w\\w+\\b', which splits "
                "or discards punctuation-laden tokens ('Cl@im', 'r3w@rds-cl@im'). Obfuscated forms like 'fr33', "
                "'g1ft', 'b0nus' are out-of-vocabulary (OOV). Because informative spam words are bypassed, "
                "the model defaults to its strong prior (86% ham) and lets the spam slip through."
            )
        },
        {
            "id": 2,
            "category": "Adversarial Spam (Polite Business / Corporate Phishing)",
            "message": "Dear valued customer, please find attached your monthly account statement and review the updated terms of service regarding your subscription at your earliest convenience.",
            "true_intent": "spam",
            "trick": "High-register corporate politeness and professional vocabulary mimicking legitimate statements.",
            "why_it_fails": (
                "Multinomial Naive Bayes assumes conditional independence of words (bag-of-words) and calculates "
                "probabilities purely from word frequencies. Corporate terms ('dear', 'valued', 'customer', "
                "'statement', 'review', 'convenience') have high posterior likelihood for 'ham' in SMS corpora. "
                "Naive Bayes possesses no semantic reasoning to detect phishing pretexts."
            )
        },
        {
            "id": 3,
            "category": "Adversarial Spam (Conversational Cloaking / Good Word Stuffing)",
            "message": "Hey bro, are we still meeting for lunch today? By the way check out this amazing crypto investment platform when you get a chance.",
            "true_intent": "spam",
            "trick": "Surrounding an unsolicited investment promotion with colloquial personal conversation.",
            "why_it_fails": (
                "Known as 'good word stuffing'. The high volume of colloquial ham tokens ('hey', 'bro', 'are', 'we', "
                "'still', 'meeting', 'lunch', 'today') generates a large cumulative ham log-likelihood that dwarfs "
                "the promo terms ('crypto', 'investment'), allowing spam to slip right through."
            )
        },
        {
            "id": 4,
            "category": "Normal Mail Flagged (Negation of Spam Keywords)",
            "message": "Not free! Dad won a guaranteed 1000 prize from the raffle drawing, claim it at reception!",
            "true_intent": "ham",
            "trick": "Negation phrase ('Not free') preceding legitimate raffle prize discussion.",
            "why_it_fails": (
                "Because bag-of-words completely ignores word order and syntactic dependency, 'Not free' is treated "
                "as two independent tokens ('not', 'free'). High-weighted spam tokens ('guaranteed', '1000', 'prize', "
                "'claim') heavily overwhelm the ham prior, triggering a False Positive."
            )
        },
        {
            "id": 5,
            "category": "Normal Mail Flagged (Urgent Family / Transactional Notice)",
            "message": "URGENT: Mom, the school contest says you won a guaranteed 1000 cash prize, call them now!",
            "true_intent": "ham",
            "trick": "Urgent family notification containing competition award vocabulary.",
            "why_it_fails": (
                "Tokens like 'URGENT', 'won', 'guaranteed', '1000', 'cash', 'prize', 'call' appear almost exclusively "
                "in spam messages in the SMS training set. MultinomialNB lacks context of family relations ('Mom', "
                "'school contest') and erroneously flags the message as spam."
            )
        },
        {
            "id": 6,
            "category": "Slide 7 Demonstration Message A",
            "message": "Cl@im your fr33 g1ft card now",
            "true_intent": "spam",
            "trick": "Obfuscated spam promise directly from Slide 7.",
            "why_it_fails": "Special characters and leetspeak bypass standard TF-IDF unigrams, predicting 'ham'."
        },
        {
            "id": 7,
            "category": "Slide 7 Demonstration Message B",
            "message": "Not free, just confirming lunch at 1pm",
            "true_intent": "ham",
            "trick": "Negation paired with everyday casual confirmation directly from Slide 7.",
            "why_it_fails": "Shows how strong casual ham words ('confirming', 'lunch', '1pm') protect against a false positive despite the word 'free'."
        }
    ]

    print("\n" + "="*80)
    print("HANDS-ON TASK: BREAKING THE SPAM FILTER RESULTS")
    print("="*80)

    results = []
    classes = list(model.classes_)

    for item in test_suite:
        text = item["message"]
        transformed = vec.transform([text])
        pred = model.predict(transformed)[0]
        probs = model.predict_proba(transformed)[0]
        prob_dict = dict(zip(classes, [round(p, 4) for p in probs]))

        is_fooled = (pred != item["true_intent"])
        status = "FOOLED FILTER" if is_fooled else "NOT FOOLED"

        result_row = {
            "ID": item["id"],
            "Category": item["category"],
            "Message": text,
            "True Label": item["true_intent"],
            "Predicted": pred,
            "Confidence": prob_dict[pred],
            "Fooled": is_fooled,
            "Failure Reason": item["why_it_fails"]
        }
        results.append(result_row)

        print(f"\n[Test #{item['id']}] {item['category']}")
        print(f"Message  : \"{text}\"")
        print(f"True Intent: {item['true_intent'].upper()} | Predicted: {pred.upper()} | Status: {status}")
        print(f"Posterior Probabilities: {prob_dict}")
        print(f"Why It Failed: {item['why_it_fails']}")

    print("\n" + "="*80)
    return results


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================
def main():
    print("="*80)
    print("AWS ML Week 03: Spam Filter Breaking & Naive Bayes Evaluation")
    print("="*80)

    # 1. Load raw dataset
    df_raw = load_raw_data()

    # 2. Execute cleaning pipeline
    print("\n--- Pipeline Step 1: Missing Values & Duplicates ---")
    df_step1 = clean_missing_and_duplicates(df_raw)

    print("\n--- Pipeline Step 2: Text Normalization ---")
    df_step2 = clean_text_fields(df_step1)

    print("\n--- Pipeline Step 3: Type Conversions ---")
    df_step3 = convert_data_types(df_step2)

    print("\n--- Pipeline Step 4: Outlier Detection ---")
    df_clean = detect_text_outliers(df_step3)

    # 3. NumPy calculations
    compute_numpy_statistics(df_clean)

    # 4. Train Slide 6 baseline model on raw dataset (reproducing slide benchmark)
    print("\n--- Training Slide 6 Model on SMS Spam Collection ---")
    vec, model, X_test, y_test = train_slide_6_model(df_raw['message'], df_raw['label'])

    # 5. Execute adversarial test suite to break filter
    run_adversarial_break_tests(vec, model)

    print("\n[SUCCESS] Week 03 pipeline and breaking tests completed successfully!")


if __name__ == "__main__":
    main()
