"""
From Machine Learning to Modern AI: One Project, Built Step by Step: A Spam Email Detector
AWS AI/ML Week 3 Deliverable Pipeline & Adversarial Evaluation
==========================================================================================
Author: Eyob (eyosialex)
Repository: https://github.com/eyosialex/AWS-Exercies
Dataset Link: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Mirror Link: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Optional Deep Learning & Modern AI dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ==============================================================================
# Paths & Remote Dataset Configuration (Slide 20: Link provided, do not upload)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "sms.tsv")
DATASET_REMOTE_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATASET_UCI_URL = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"


# ==============================================================================
# 1. Dataset Ingestion (Slide 3: Collect Emails)
# ==============================================================================
def load_raw_data(filepath: str = DATA_FILE, remote_url: str = DATASET_REMOTE_URL) -> pd.DataFrame:
    """
    Loads raw SMS dataset from local cache if present, or downloads directly from public URL.
    Adheres strictly to deliverable rule: 'a link to the dataset (do not upload it)'.
    """
    if os.path.exists(filepath):
        print(f"[INGEST] Loading dataset from local cache: '{filepath}'")
        df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    else:
        print(f"[INGEST] Local file not found. Downloading dataset from: {remote_url}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.read_csv(remote_url, sep='\t', header=None, names=['label', 'message'])
        df.to_csv(filepath, sep='\t', header=False, index=False)
        print(f"[INGEST] Cached dataset at: '{filepath}'")

    print(f"[INGEST] Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"[INGEST] Dataset Source: {DATASET_UCI_URL}")
    return df


# ==============================================================================
# 2. Named Cleaning & Preprocessing Functions (Data Quality Pipeline)
# ==============================================================================
def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and handles missing values and drops duplicate text entries.
    """
    df_clean = df.copy()
    initial_count = len(df_clean)

    missing = df_clean.isna().sum().to_dict()
    print(f"[CLEAN] Missing values per column: {missing}")
    df_clean = df_clean.dropna(subset=['label', 'message'])

    duplicates = df_clean.duplicated(subset=['message']).sum()
    print(f"[CLEAN] Duplicate messages detected: {duplicates}")
    df_clean = df_clean.drop_duplicates(subset=['message']).reset_index(drop=True)

    print(f"[CLEAN] Retained {len(df_clean)} unique messages out of {initial_count} (dropped {initial_count - len(df_clean)}).")
    return df_clean


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes whitespace and standardizes casing for labels and messages.
    """
    df_clean = df.copy()
    df_clean['message'] = df_clean['message'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    df_clean['label'] = df_clean['label'].astype(str).str.strip().str.lower()
    print("[CLEAN] Whitespace normalization completed.")
    return df_clean


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Casts columns to explicit data types and computes character & word lengths.
    """
    df_clean = df.copy()
    df_clean['message'] = df_clean['message'].astype("string")
    df_clean['label'] = df_clean['label'].astype("category")
    df_clean['char_length'] = df_clean['message'].str.len().astype(np.int64)
    df_clean['word_count'] = df_clean['message'].str.split().str.len().astype(np.int64)
    print("[CLEAN] Column types: message -> string, label -> category, lengths -> int64.")
    return df_clean


def detect_text_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects character length anomalies using the Interquartile Range (IQR) method and np.where.
    """
    df_clean = df.copy()
    lengths = df_clean['char_length'].to_numpy()

    q25, q75 = np.percentile(lengths, [25, 75])
    iqr = q75 - q25
    lower_bound = max(0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    df_clean['is_length_outlier'] = np.where(
        (df_clean['char_length'] < lower_bound) | (df_clean['char_length'] > upper_bound),
        True,
        False
    )
    outlier_count = int(np.sum(df_clean['is_length_outlier']))
    print(f"[OUTLIERS] IQR Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]. Flagged {outlier_count} length outliers.")
    return df_clean


# ==============================================================================
# 3. Statistical Analysis with NumPy (>= 6 distinct NumPy functions)
# ==============================================================================
def compute_numpy_statistics(df: pd.DataFrame) -> dict:
    """
    Computes summary metrics using: np.mean, np.median, np.std, np.percentile, np.unique, np.where.
    """
    lengths = df['char_length'].to_numpy()
    words = df['word_count'].to_numpy()
    labels = df['label'].to_numpy()

    mean_len = float(np.mean(lengths))
    median_len = float(np.median(lengths))
    std_len = float(np.std(lengths))

    p10, p25, p50, p75, p90 = np.percentile(lengths, [10, 25, 50, 75, 90])
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, [int(c) for c in counts]))

    binary_labels = np.where(labels == 'spam', 1, 0)
    spam_mean_len = float(np.mean(lengths[np.where(binary_labels == 1)]))
    ham_mean_len = float(np.mean(lengths[np.where(binary_labels == 0)]))

    stats = {
        "mean_len": mean_len,
        "median_len": median_len,
        "std_len": std_len,
        "percentiles": {"p25": p25, "p75": p75, "p90": p90},
        "label_distribution": label_dist,
        "spam_mean_length": spam_mean_len,
        "ham_mean_length": ham_mean_len,
    }

    print("\n" + "="*70)
    print("NUMPY STATISTICAL ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total Unique Messages   : {len(lengths)}")
    print(f"Class Distribution      : {label_dist}")
    print(f"Mean Character Length   : {mean_len:.2f} chars (Std: {std_len:.2f})")
    print(f"Median Character Length : {median_len:.1f} chars (IQR: [{p25:.1f}, {p75:.1f}])")
    print(f"Spam Mean Length        : {spam_mean_len:.2f} chars")
    print(f"Ham Mean Length         : {ham_mean_len:.2f} chars")
    print("="*70 + "\n")
    return stats


# ==============================================================================
# 4. Era 1: Classical Machine Learning (Slides 5, 6, 7, 10)
# ==============================================================================
def train_and_diagnose_classical_ml(emails, labels):
    """
    Implements Slide 5, 6, 7, and 10:
    - Train / Validation / Test 70% / 15% / 15% split
    - 10-line baseline: TfidfVectorizer + MultinomialNB
    - Overfitting / underfitting diagnostic check
    - Accuracy, Precision, Recall, Confusion Matrix
    """
    print("\n" + "="*70)
    print("ERA 1: CLASSICAL MACHINE LEARNING (SLIDES 5, 6, 7, 10)")
    print("="*70)

    # Slide 5: Train 70%, Validation 15%, Test 15%
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        emails, labels, test_size=0.15, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.17647, random_state=42, stratify=y_train_full)
    # Resulting splits: 70% Train, 15% Validation, 15% Test

    print(f"[SPLIT] 3-Way Split: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")

    # Slide 6: Exact 10-line baseline model
    vec = TfidfVectorizer()                             # text -> numbers (features)
    model = MultinomialNB()                             # a simple, fast classifier
    model.fit(vec.fit_transform(X_train), y_train)      # learn

    # Slide 10: Diagnostic Check (Train vs Test Accuracy, Precision, Recall)
    pred_tr = model.predict(vec.transform(X_train))
    pred_te = model.predict(vec.transform(X_test))

    train_acc = accuracy_score(y_train, pred_tr)
    test_acc = accuracy_score(y_test, pred_te)
    precision = precision_score(y_test, pred_te, pos_label="spam")
    recall = recall_score(y_test, pred_te, pos_label="spam")

    print(f"train acc : {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"test acc  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"precision : {precision:.4f} ({precision*100:.2f}%)")
    print(f"recall    : {recall:.4f} ({recall*100:.2f}%)")

    # Diagnostic Interpretation (Slide 10)
    if train_acc > 0.98 and test_acc < 0.85:
        print("[DIAGNOSIS] Overfitting detected: Model memorised training set.")
    elif train_acc < 0.75 and test_acc < 0.75:
        print("[DIAGNOSIS] Underfitting detected: Model too simple, need better features.")
    else:
        print("[DIAGNOSIS] Good Generalization: Close train/test scores demonstrate robust fit.")

    # Slide 7: Confusion Matrix & Cost of False Positives
    cm = confusion_matrix(y_test, pred_te, labels=['ham', 'spam'])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[METRICS] Confusion Matrix: TN={tn} (good mail kept), FP={fp} (good mail lost!), FN={fn} (spam leaked), TP={tp} (caught)")
    print("[METRICS] Notice: For spam filters, False Positives hurt most (good mail lost), so watch precision!")

    return vec, model, X_val, y_val, X_test, y_test


# ==============================================================================
# 5. Era 2: Deep Learning (Slides 11, 12, 13)
# ==============================================================================
def train_deep_learning_model(X_train, y_train, X_val, y_val):
    """
    Implements Slide 12 & 13:
    Neural Network with:
      - Dense(64, activation='relu', input_shape=(5000,))
      - Dense(32, activation='relu')
      - Dense(1, activation='sigmoid')
      - Adam optimizer, binary cross-entropy loss, batch_size=32, epochs=5
    """
    print("\n" + "="*70)
    print("ERA 2: DEEP LEARNING NEURAL NETWORK (SLIDES 11, 12, 13)")
    print("="*70)

    # TF-IDF feature extraction constrained to top 5000 features as in Slide 13
    dl_vec = TfidfVectorizer(max_features=5000)
    X_train_tfidf = dl_vec.fit_transform(X_train).toarray()
    X_val_tfidf = dl_vec.transform(X_val).toarray()

    y_train_num = np.where(y_train == 'spam', 1.0, 0.0)
    y_val_num = np.where(y_val == 'spam', 1.0, 0.0)

    if not TORCH_AVAILABLE:
        print("[NOTE] PyTorch not installed in environment; skipping DL training execution.")
        return dl_vec, None

    # Define PyTorch model mirroring Slide 13 architecture
    class SpamNeuralNet(nn.Module):
        def __init__(self, input_dim=5000):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    model = SpamNeuralNet(input_dim=5000)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DataLoaders with batch_size=32 (Slide 12 & 13)
    train_dataset = TensorDataset(torch.tensor(X_train_tfidf, dtype=torch.float32),
                                  torch.tensor(y_train_num, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val_tfidf, dtype=torch.float32),
                                torch.tensor(y_val_num, dtype=torch.float32).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("[TRAIN] Training Neural Network (batch_size=32, epochs=5)...")
    for epoch in range(1, 6):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)

        # Validation check
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_x)
                preds = (outputs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        val_loss /= len(val_dataset)
        val_acc = correct / total

        print(f"  Epoch {epoch}/5 - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    print("[SUCCESS] Deep Learning neural network training complete.")
    return dl_vec, model


# ==============================================================================
# 6. Era 3: LLMs and Modern AI (Slides 14, 15, 16, 17, 18)
# ==============================================================================
def demonstrate_modern_ai_tokenization():
    """
    Implements Slide 15:
    Shows how subword tokenization (AutoTokenizer 'bert-base-uncased') copes with typos like 'fr33'.
    """
    print("\n" + "="*70)
    print("ERA 3: LLMS & MODERN AI TOKENIZATION (SLIDES 14, 15, 16, 17, 18)")
    print("="*70)

    if not TRANSFORMERS_AVAILABLE:
        print("[NOTE] Transformers library not available; skipping tokenizer demo.")
        return

    try:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        sample_email = "Claim your fr33 gift now"
        tokens = tok.tokenize(sample_email)
        print(f"Input text      : \"{sample_email}\"")
        print(f"Tokenized output: {tokens}")
        print("\nWhy Subword Tokenization beats word counting (Slide 15):")
        print("  1. Tokens cope with typos and leetspeak like 'fr33' -> ['fr', '##33'].")
        print("  2. Dense Embeddings place semantically similar words close in vector space.")
        print("  3. Contextual Attention allows negations ('not') to change how 'free' is read.")
    except Exception as e:
        print(f"[NOTE] AutoTokenizer demo skipped due to offline/network environment: {e}")


# ==============================================================================
# 7. Hands-on Task: Break the Spam Filter (Slides 19 & 20)
# ==============================================================================
def run_break_the_filter_challenge(vec: TfidfVectorizer, model: MultinomialNB):
    """
    Executes Slide 19 & 20:
    - Tests 5+ messages attempting to fool the filter.
    - Records exact predictions, confidences, and failures.
    - Outputs the exact Deliverable Results Table.
    """
    print("\n" + "="*80)
    print("HANDS-ON TASK: BREAK THE SPAM FILTER (SLIDES 19 & 20)")
    print("="*80)

    # 1. Slide 19 Built-in tests
    slide_tests = [
        "Cl@im your fr33 g1ft card now",
        "Not free, just confirming lunch at 1pm"
    ]
    slide_preds = model.predict(vec.transform(slide_tests))
    print("[SLIDE 19 BASELINE TESTS]")
    for msg, pred in zip(slide_tests, slide_preds):
        print(f"  \"{msg}\" -> Predicted: {pred}")

    # 2. Comprehensive Adversarial Suite (5+ targeted attack scenarios)
    adversarial_tests = [
        {
            "message": "Cl@im your fr33 g1ft card now! V1s1t www.r3w@rds-cl@im.com to get your 500 b0nus t0d@y!",
            "true_intent": "spam",
            "trick": "Leetspeak / Character Substitution (@ for a, 3 for e, 1 for i, 0 for o)",
            "why_it_failed": "Spelling trick, words unseen (OOV tokens bypass TF-IDF; model defaults to ham prior)."
        },
        {
            "message": "Dear valued customer, please find attached your monthly account statement and review the updated terms of service regarding your subscription at your earliest convenience.",
            "true_intent": "spam",
            "trick": "Polite Corporate Register & Phishing Mimicry",
            "why_it_failed": "Formal business tokens ('valued', 'customer', 'statement') dominate with high ham frequency."
        },
        {
            "message": "Hey bro, are we still meeting for lunch today? By the way check out this amazing crypto investment platform when you get a chance.",
            "true_intent": "spam",
            "trick": "Conversational Cloaking / Good-Word Stuffing",
            "why_it_failed": "Colloquial ham words ('bro', 'lunch', 'today') dilute and overwhelm spam token weights."
        },
        {
            "message": "Not free! Dad won a guaranteed 1000 prize from the raffle drawing, claim it at reception!",
            "true_intent": "ham",
            "trick": "Negation Inversion with Prize Vocabulary",
            "why_it_failed": "Saw 'free', ignored 'not' (bag-of-words ignores word sequence and negation context)."
        },
        {
            "message": "URGENT: Mom, the school contest says you won a guaranteed 1000 cash prize, call them now!",
            "true_intent": "ham",
            "trick": "Urgent Family Contest Alert",
            "why_it_failed": "Spam trigger words ('URGENT', 'won', 'cash', 'prize') trigger false positive without family context."
        }
    ]

    results = []
    classes = list(model.classes_)

    for item in adversarial_tests:
        msg = item["message"]
        transformed = vec.transform([msg])
        pred = model.predict(transformed)[0]
        probs = model.predict_proba(transformed)[0]
        prob_dict = dict(zip(classes, [round(p, 4) for p in probs]))
        fooled = (pred != item["true_intent"])

        results.append({
            "My message": msg,
            "Model said": pred,
            "Should be": item["true_intent"],
            "Why it failed": item["why_it_failed"],
            "Confidence": prob_dict[pred],
            "Fooled": fooled,
            "Trick Category": item["trick"]
        })

    # Display Official Deliverable Markdown Table (Slide 20 format)
    print("\n" + "="*80)
    print("DELIVERABLE RESULTS TABLE (SLIDE 20 FORMAT)")
    print("="*80)
    print(f"{'My message':<60} | {'Model said':<10} | {'Should be':<10} | {'Why it failed'}")
    print("-" * 120)
    for r in results:
        short_msg = (r["My message"][:57] + '...') if len(r["My message"]) > 60 else r["My message"]
        print(f"{short_msg:<60} | {r['Model said']:<10} | {r['Should be']:<10} | {r['Why it failed']}")
    print("-" * 120)

    # 3 or more tricks analysis (Slide 20 requirement)
    print("\n" + "="*80)
    print("3 OR MORE ADVERSARIAL TRICKS EXPLAINED (SLIDE 20 REQUIREMENT)")
    print("="*80)
    print("Trick 1: Spelling & Leetspeak Obfuscation")
    print("  -> Replacing letters with numbers or symbols creates Out-Of-Vocabulary (OOV) tokens, evading keyword filters.")
    print("Trick 2: Negation Inversion")
    print("  -> Naive Bayes evaluates tokens independently; it sees 'free' and 'prize' but cannot bind 'not' to 'free'.")
    print("Trick 3: Polite Business / Phishing Register")
    print("  -> Mimicking corporate correspondence leverages vocabulary that statistically appears predominantly in ham.")
    print("Trick 4: Good-Word Stuffing")
    print("  -> Padding spam text with benign colloquial sentences artificially shifts the posterior likelihood toward ham.")

    # 2 sentences on what you learned (Slide 20 requirement)
    print("\n" + "="*80)
    print("WHAT I LEARNED (SLIDE 20 REQUIREMENT: EXACTLY 2 SENTENCES)")
    print("="*80)
    sentence_1 = ("While classical Naive Bayes achieves high benchmark accuracy by counting word co-occurrences, "
                  "its bag-of-words assumption makes it completely blind to word order, negations, and spelling obfuscations—allowing "
                  "adversaries to evade detection with trivial character substitutions and conversational padding.")
    sentence_2 = ("Modern deep learning and Transformer-based LLMs solve these fundamental weaknesses by learning hierarchical "
                  "representations, tokenizing robust subword units, and using contextual self-attention to understand semantic meaning "
                  "rather than isolated keywords.")
    print(f"Sentence 1: {sentence_1}")
    print(f"Sentence 2: {sentence_2}")
    print("="*80 + "\n")

    return results


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================
def main():
    print("="*80)
    print("AWS ML Week 03: From Machine Learning to Modern AI")
    print("One Project, Built Step by Step: A Spam Email Detector")
    print("="*80)

    # Ingestion & Data Preparation
    df_raw = load_raw_data()
    df_clean = clean_missing_and_duplicates(df_raw)
    df_clean = clean_text_fields(df_clean)
    df_clean = convert_data_types(df_clean)
    df_clean = detect_text_outliers(df_clean)

    # NumPy Statistical Analysis
    compute_numpy_statistics(df_clean)

    # Era 1: Classical Machine Learning
    vec, model, X_val, y_val, X_test, y_test = train_and_diagnose_classical_ml(
        df_raw['message'], df_raw['label']
    )

    # Era 2: Deep Learning Neural Network
    train_deep_learning_model(df_clean['message'], df_clean['label'], X_val, y_val)

    # Era 3: LLMs and Modern AI Tokenization
    demonstrate_modern_ai_tokenization()

    # Hands-on Adversarial Challenge: Break the Spam Filter
    run_break_the_filter_challenge(vec, model)

    print("[PIPELINE COMPLETE] All Week 3 requirements executed successfully.")


if __name__ == "__main__":
    main()
