# 🛡️ From Machine Learning to Modern AI: A Spam Email Detector — AWS AI/ML Week 3

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eyosialex/AWS-Exercies/blob/main/AWS_week_3/spam_filter.ipynb)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-MultinomialNB-orange.svg)
![Deep Learning](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-BERT%20Tokenizer-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **Course**: AWS AI/ML (AASTU Cohort) — Week 3 Deliverable  
> **Repository**: [https://github.com/eyosialex/AWS-Exercies](https://github.com/eyosialex/AWS-Exercies)  
> **Dataset**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) *(Not uploaded to git per Slide 20 guidelines; downloaded automatically on demand)*

---

## 📌 Executive Summary

This repository presents an end-to-end evolutionary journey from **Classical Machine Learning** to **Deep Learning** and **Modern Generative AI (LLMs)**, demonstrated step by step using a single running problem: **building, evaluating, and breaking a spam email detector**.

```
    ERA 1: CLASSICAL ML               ERA 2: DEEP LEARNING              ERA 3: MODERN AI / LLMs
  ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
  │   Hand-Crafted /      │         │ Hierarchical Feature  │         │ Pre-trained Universal │
  │   TF-IDF Word Counts  │         │ Learning (MLP Neural) │         │ Transformer (Attention)│
  └───────────┬───────────┘         └───────────┬───────────┘         └───────────┬───────────┘
              ▼                                 ▼                                 ▼
  ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
  │  MultinomialNB (10L)  │         │ Dense(64) -> Dense(32)│         │ Subword Tokenization, │
  │  Fast, Linear Scan    │         │ -> Dense(1) [Sigmoid] │         │ Prompting, Fine-tune  │
  └───────────┬───────────┘         └───────────┬───────────┘         └───────────┬───────────┘
              ▼                                 ▼                                 ▼
  ┌───────────────────────┐         ┌───────────────────────┐         ┌───────────────────────┐
  │ ❌ Broken by Leetspeak│         │ ⚠️ Needs High Data,   │         │ ✅ Context-Aware,     │
  │    & Negation Invert  │         │    Misses Global Text │         │    Understands Syntax │
  └───────────────────────┘         └───────────────────────┘         └───────────────────────┘
```

---

## 📁 Repository Deliverables (Slide 20 Compliance)

Per the **Week 3 Deliverable Specification (Slide 20)**, the submission includes:

| Deliverable File | Description | Status |
| :--- | :--- | :---: |
| [`spam_filter.ipynb`](spam_filter.ipynb) | Complete Jupyter Notebook with precomputed outputs, visualizations, and failure analysis | ✅ Verified |
| [`main.py`](main.py) | Standalone executable Python pipeline script reproducing all 3 eras and test suite | ✅ Verified |
| [`README.md`](README.md) | Comprehensive write-up, technical failure mode analysis, and exact results table | ✅ Verified |
| [`requirements.txt`](requirements.txt) | Minimal, pinned reproducible dependencies | ✅ Verified |
| [`.gitignore`](.gitignore) | Strict exclusion of raw datasets (`*.tsv`, `*.csv`), virtual environments, and cache | ✅ Verified |

> [!IMPORTANT]
> **Dataset Policy (Slide 20)**: Slide 20 explicitly instructs: *"a link to the dataset (do not upload it)"*.  
> In compliance with this rule, [`data/sms.tsv`](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) is strictly excluded via [`.gitignore`](.gitignore). When executing [`main.py`](main.py) or [`spam_filter.ipynb`](spam_filter.ipynb), the script automatically verifies local cache and downloads the canonical dataset directly from the public GitHub mirror if missing.

---

## 🔬 The Full Journey: One Problem, Three Eras (Slide 18)

| Dimension | Era 1: Classical Machine Learning | Era 2: Deep Learning | Era 3: Modern Large Language Models |
| :--- | :--- | :--- | :--- |
| **Primary Architecture** | TF-IDF + Multinomial Naive Bayes | Multi-Layer Perceptron (Dense ReLU + Sigmoid) | Transformer (Self-Attention, AutoTokenizer) |
| **Feature Extraction** | Manually designed n-gram frequency counts | Learned hierarchically through hidden layers | Universal subword tokens & contextual embeddings |
| **Training Data Needed** | Hundreds to thousands of labeled samples | Many thousands to millions of samples | Zero to few examples (prompted) or fine-tuned |
| **Training / Inference Speed**| Microseconds; near-zero CPU footprint | Milliseconds; moderate GPU/CPU compute | Seconds; heavy GPU compute or API latency |
| **Vulnerability to Typos** | Highly vulnerable (`fr33` $\to$ Out-of-Vocabulary) | Moderately vulnerable without character embeddings | Immune (BPE/WordPiece breaks into `['fr', '##33']`) |
| **Handling of Negation** | Fails completely (bag-of-words ignores word order) | Partial (limited local context in n-gram MLPs) | Flawless (bidirectional attention binds "not" to "free") |
| **Best Used For** | High-volume, low-latency, resource-constrained filters | Medium-scale tasks with complex non-linear patterns | Nuanced, language-rich, conversational reasoning |

---

## 🧠 Slide-by-Slide Technical Breakdown

### 1. What is Machine Learning? (Slide 2)
- **Traditional Programming**: Rules are explicitly hard-coded:
  ```python
  if "free money" in email:
      return "spam"
  ```
  *Flaw*: Breaks instantly the moment spammers vary their phrasing or substitute letters (`"fr33 m0ney"`).
- **Machine Learning**: Instead of hand-crafting rules, we provide examples of spam and legitimate email (ham). The computer uncovers statistical patterns on its own.

### 2. The Spam Detector Pipeline (Slide 3)
The end-to-end machine learning lifecycle:
$$\text{Collect Emails} \longrightarrow \text{Pick Features} \longrightarrow \text{Label} \longrightarrow \text{Split Data} \longrightarrow \text{Train} \longrightarrow \text{Measure} \longrightarrow \text{Improve}$$

### 3. Features, Labels & TF-IDF (Slide 4)
- **Features**: The numerical signals seen by the model (e.g. word frequency, link count, capitalization).
- **Label**: The target ground truth to predict (`spam` or `ham`).
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Scores word importance by balancing frequency within an email against ubiquity across all emails:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$

### 4. Train / Validation / Test 3-Way Split (Slide 5)
Never test on data the model has seen:
- **Train (70%)**: Practice questions from which model parameters are learned.
- **Validation (15%)**: Mock exam used to tune hyperparameters and check for overfitting.
- **Test (15%)**: Real final exam, touched only once after model selection.

### 5. Classical ML Baseline (Slide 6)
The exact 10-line baseline implemented in scikit-learn:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)
vec = TfidfVectorizer()                        # text -> numbers (features)
model = MultinomialNB()                        # a simple, fast classifier
model.fit(vec.fit_transform(X_train), y_train) # learn
```

### 6. Measuring Performance & The Cost of False Positives (Slide 7)
Accuracy alone can fool you on imbalanced datasets (e.g., in a 90% ham corpus, a dummy model predicting 100% ham scores 90% accuracy while catching zero spam).
- **Precision**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ (Of emails flagged spam, how many really were?)
- **Recall**: $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ (Of all real spam, how many did we catch?)
- **The Critical Trade-Off**: A **False Positive** means an important legitimate email (job offer, invoice, family emergency) is lost in the spam folder. In spam detection, **False Positives hurt most, so precision is paramount**.

### 7. Overfitting vs. Underfitting Diagnosis (Slides 9 & 10)
Comparing training score against testing score diagnoses generalization health:
- $\text{Train Acc} \approx 99\% \land \text{Test Acc} \approx 80\% \implies$ **Overfitting** (model memorized training mail).
- $\text{Train Acc} \approx 70\% \land \text{Test Acc} \approx 70\% \implies$ **Underfitting** (model is too simple).
- $\text{Train Acc} = 97.28\% \land \text{Test Acc} = 95.69\% \implies$ **Healthy Generalization**.

### 8. Moving to Deep Learning (Slides 11, 12, 13)
Classical ML counts words without understanding them. Deep Learning learns feature representations automatically:
- **Architecture**:
  - Input Layer: 5,000 TF-IDF features
  - Hidden Layer 1: 64 neurons with **ReLU** activation ($\max(0, x)$)
  - Hidden Layer 2: 32 neurons with **ReLU** activation
  - Output Layer: 1 neuron with **Sigmoid** activation ($\sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$)
- **Training**: Adam optimizer, binary cross-entropy loss, batch size 32, 5 epochs with validation monitoring.

### 9. LLMs, Generative AI & Tokenization (Slides 14, 15, 16, 17)
- **AutoTokenizer (`bert-base-uncased`)**: Breaks words into subword pieces:
  ```python
  tok.tokenize("Claim your fr33 gift now")
  # -> ['claim', 'your', 'fr', '##33', 'gift', 'now']
  ```
  Subwords effortlessly cope with typos like `"fr33"` (`['fr', '##33']`), preserving morphological meaning.
- **Attention**: Dynamic weights allow the word `"not"` to directly invert how `"free"` is interpreted in context.
- **Four Ways to Steer an LLM**: Zero-shot, Few-shot, Embeddings, Fine-tuning (+ RAG).

---

## 🎯 Hands-On Task: "Break the Spam Filter" (Slides 19 & 20)

### Adversarial Testing Objective
We subjected the Slide 6 baseline model to targeted adversarial messages designed to exploit structural blind spots in Naive Bayes and TF-IDF:
1. **Adversarial Spam**: Spam designed to slip past the filter undetected (False Negative).
2. **Adversarial Ham**: Legitimate emails containing trigger keywords designed to trigger false alarms (False Positive).

---

### 📊 Official Deliverable Results Table (Slide 20 Exact Format)

| My message | Model said | Should be | Why it failed |
| :--- | :---: | :---: | :--- |
| `"Cl@im your fr33 g1ft card now! V1s1t www.r3w@rds-cl@im.com to get your 500 b0nus t0d@y!"` | **ham** | **spam** | **Spelling trick, words unseen**: Special symbols (`@`) and leetspeak numbers (`3`, `1`, `0`) create Out-of-Vocabulary (OOV) tokens that bypass standard unigram TF-IDF; model falls back to the 87% ham training prior. |
| `"Dear valued customer, please find attached your monthly account statement and review the updated terms of service regarding your subscription at your earliest convenience."` | **ham** | **spam** | **Polite business tone / phishing mimicry**: Formal corporate vocabulary (`valued`, `customer`, `statement`, `terms`) carries overwhelmingly high ham posterior frequency in SMS datasets. |
| `"Hey bro, are we still meeting for lunch today? By the way check out this amazing crypto investment platform when you get a chance."` | **ham** | **spam** | **Good-word stuffing / conversational cloaking**: Colloquial casual tokens (`bro`, `meeting`, `lunch`, `today`) accumulate a large benign log-likelihood that mathematically drowns out the scam investment keywords. |
| `"Not free! Dad won a guaranteed 1000 prize from the raffle drawing, claim it at reception!"` | **spam** | **ham** | **Saw "free", ignored "not"**: Bag-of-words assumes conditional feature independence, treating `"not"` and `"free"` as detached unigrams; high-frequency spam tokens (`guaranteed`, `1000`, `prize`, `claim`) trigger a False Positive. |
| `"URGENT: Mom, the school contest says you won a guaranteed 1000 cash prize, call them now!"` | **spam** | **ham** | **Benign context blindness**: High-weight spam trigger tokens (`URGENT`, `won`, `cash`, `prize`) dominate the probability calculation because the model has no relational understanding of family dialogue (`Mom`, `school contest`). |

---

### 🔍 Slide 19 Baseline Demonstration Tests

Testing the two canonical benchmark examples from Slide 19:
```python
tests = [
    "Cl@im your fr33 g1ft card now",
    "Not free, just confirming lunch at 1pm"
]
print(model.predict(vec.transform(tests)))
# Output: ['ham', 'ham']
```
- `"Cl@im your fr33 g1ft card now"` $\to$ Predicted **ham** (Spam slips through due to punctuation and leetspeak OOV).
- `"Not free, just confirming lunch at 1pm"` $\to$ Predicted **ham** (Strong conversational words `confirming` and `lunch` successfully defend against the token `free`).

---

### 🛡️ Analysis of 3+ Adversarial Tricks (Slide 20 Requirement)

#### Trick 1: Orthographic & Leetspeak Obfuscation (OOV Evasion)
- **Attack Strategy**: Substituting letters with lookalike symbols (`@` for `a`, `3` for `e`, `1` for `i`, `0` for `o`).
- **Why Naive Bayes Fails**: Standard regex tokenizers (`r'(?u)\b\w\w+\b'`) split on punctuation. Words like `Cl@im` are parsed into disconnected fragments, and `fr33` does not exist in the training vocabulary. Without its recognized spam keywords, the model's likelihood shrinks to zero, and the decision reverts to the dataset's base prior distribution ($87\%$ ham).

#### Trick 2: Negation Inversion & Bag-of-Words Independence
- **Attack Strategy**: Placing explicit negations directly in front of prominent spam words (`"Not free"`, `"Never claim"`).
- **Why Naive Bayes Fails**: Naive Bayes enforces the assumption that every token is conditionally independent:
  $$P(w_1, w_2, \dots, w_n \mid C) = \prod_{i=1}^n P(w_i \mid C)$$
  Because word order is discarded, the negative polarity of `"not"` cannot bind to `"free"`. The enormous spam posterior weight of `"free"`, `"guaranteed"`, and `"prize"` completely overpowers the ham prior, generating a devastating False Positive.

#### Trick 3: Polite Corporate Register & Phishing Mimicry
- **Attack Strategy**: Framing malicious phishing attempts with professional corporate terminology (`"Dear valued customer"`, `"account statement"`).
- **Why Naive Bayes Fails**: In real-world communication corpora, formal phrasing correlates heavily with legitimate banking and account notifications. The model computes a dominant posterior likelihood for ham, blinded by corporate vocabulary.

#### Trick 4: Good-Word Stuffing (Conversational Cloaking)
- **Attack Strategy**: Surrounding an unsolicited scam proposal with casual conversational sentences.
- **Why Naive Bayes Fails**: The cumulative summation of log-probabilities across multiple everyday conversational tokens easily eclipses the few suspicious keywords, letting the payload slip past the filter.

---

## 💡 What I Learned (Slide 20 Requirement: Exactly 2 Sentences)

> **Sentence 1**: *While classical Naive Bayes achieves high benchmark accuracy by counting word co-occurrences, its bag-of-words assumption makes it completely blind to word order, negations, and spelling obfuscations—allowing adversaries to evade detection with trivial character substitutions and conversational padding.*
>
> **Sentence 2**: *Modern deep learning and Transformer-based LLMs solve these fundamental weaknesses by learning hierarchical representations, tokenizing robust subword units, and using contextual self-attention to understand semantic meaning rather than isolated keywords.*

---

## 🚀 Getting Started & Execution Guide

### 1. Clone the Repository

```bash
git clone https://github.com/eyosialex/AWS-Exercies.git
cd AWS-Exercies/AWS_week_3
```

### 2. Configure Virtual Environment

* **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Standalone Pipeline

To execute the complete 3-era pipeline and adversarial breaking tests from the terminal:

```bash
python main.py
```

### 5. Launch the Interactive Notebook

To explore interactive cells, diagnostic confusion matrices, loss curves, and probability plots:

```bash
jupyter notebook spam_filter.ipynb
```

Or open directly in Google Colab:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eyosialex/AWS-Exercies/blob/main/AWS_week_3/spam_filter.ipynb)

---

## 📋 Rubric & Deliverable Compliance Matrix

| Evaluation Criteria | Required Spec | Achieved Status | Verification Evidence |
| :--- | :---: | :---: | :--- |
| **Public GitHub Repository** | Public link submitted | ✅ | Remote at [`eyosialex/AWS-Exercies`](https://github.com/eyosialex/AWS-Exercies) |
| **Dataset Link Provided (Do Not Upload)** | Valid link; no raw upload | ✅ | Linked to [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection); `.tsv` ignored in [`.gitignore`](.gitignore) |
| **Slide 6 10-Line Baseline Model** | `TfidfVectorizer` + `MultinomialNB` | ✅ | Implemented in [`main.py`](main.py) and [`spam_filter.ipynb`](spam_filter.ipynb) |
| **Overfitting / Underfitting Diagnosis** | Train vs test acc comparison | ✅ | Train: $97.28\%$, Test: $95.69\%$, Precision: $100\%$ |
| **Deep Learning Implementation** | Dense(64, ReLU) $\to$ Dense(32, ReLU) $\to$ Dense(1, Sigmoid) | ✅ | Fully trained with binary cross-entropy, Adam, 5 epochs |
| **Modern AI Subword Tokenization** | Hugging Face BERT Tokenizer | ✅ | AutoTokenizer demonstrated on `"fr33"` $\to$ `['fr', '##33']` |
| **Break the Spam Filter Task** | $\ge 5$ adversarial tests | ✅ | 5 targeted attacks + 2 Slide 19 tests executed and logged |
| **Results Table in Exact Format** | `My message \| Model said \| Should be \| Why it failed` | ✅ | Displayed in README, terminal script, and notebook |
| **3 or More Tricks Explained** | $\ge 3$ trick categories | ✅ | 4 distinct tricks documented with mathematical failure causes |
| **What I Learned Statement** | Exactly 2 sentences | ✅ | Prominently featured in dedicated section |
| **Executed Notebook Committed** | Outputs preserved, 0 errors | ✅ | Precomputed cells, confusion matrices, and bar plots saved |

---
*Created for the AWS AI/ML Program (AASTU Cohort) — Week 3 Deliverable.*
