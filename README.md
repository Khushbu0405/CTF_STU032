# CTF Challenge - Book Review Authenticity Analysis

A Capture The Flag (CTF) challenge involving detection of manipulated book reviews using data analysis, machine learning, and SHAP explainability.

---

## Challenge Overview

This CTF involves analyzing a dataset of books and reviews to:
1. Identify a specific manipulated book using hash-based clues
2. Detect a fake review containing a planted identifier
3. Use ML and SHAP to determine what makes reviews appear genuine vs suspicious

**Student ID**: STU032  
---

## Quick Start

### Prerequisites
```bash
pip install pandas scikit-learn shap numpy
```

### Files Needed
- `books.csv` - Book dataset with ratings and metadata
- `reviews.csv` - Review dataset with text and ratings
- `solver.py` - Main solution script

### Run the Solution
```bash
python solver.py
```

### Output
Creates `flags.txt` containing:
```
FLAG1 = 542523382ad4d0e10d6eacd47356e5decab43c238cc29c1d60f3bb18ebc0fbb4
FLAG2 = FLAG2{F853BFAD}
FLAG3 = FLAG3{5412dec5fa}
```

---

## Flags Captured

| Flag | Value | Method |
|------|-------|--------|
| **FLAG1** | `542523382ad4d0e10d6eacd47356e5decab43c238cc29c1d60f3bb18ebc0fbb4` | Book title hash extraction |
| **FLAG2** | `FLAG2{F853BFAD}` | Fake review identification |
| **FLAG3** | `FLAG3{5412dec5fa}` | ML + SHAP analysis |

---

## Solution Approach

### Step 1: FLAG1 - Find Your Book
1. Compute hash: `SHA256("STU032")` → `F853BFAD` (first 8 chars)
2. Filter books with `rating_number=1234` and `average_rating=5.0`
3. Scan reviews for the hash `F853BFAD`
4. Match review ASIN to filtered books → "Tom Jones - The Life"
5. Extract first 8 non-space chars from title → "TomJones"
6. Compute `SHA256("TomJones")` → FLAG1

### Step 2: FLAG2 - Identify Fake Review
1. The review containing hash `F853BFAD` is the fake review
2. FLAG2 is simply the hash itself: `FLAG2{F853BFAD}`

### Step 3: FLAG3 - SHAP Analysis
1. **Label Reviews**: Create suspicious vs genuine labels
   - Suspicious: Short + generic praise words
   - Genuine: Longer + book-specific terms
2. **Train Model**: Use Logistic Regression on TF-IDF features
3. **SHAP Analysis**: Identify top 3 words that reduce suspicion
   - Found: "good", "have", "if"
4. **Compute FLAG3**: 
   - Concatenate: `goodhaveif032`
   - SHA256 hash → first 10 chars → `5412dec5fa`
   - Format: `FLAG3{5412dec5fa}`

---

## Technical Details

### Technologies Used
- **Python 3.11**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning (TF-IDF, Logistic Regression)
- **SHAP** - Model explainability
- **hashlib** - Cryptographic hashing

### Key Algorithms
- **TF-IDF Vectorization**: Convert text reviews to numerical features
- **Logistic Regression**: Binary classification (suspicious vs genuine)
- **SHAP Linear Explainer**: Identify feature contributions to predictions
- **SHA256 Hashing**: Generate flags from computed strings

### Data Processing
```python
# Review Labeling Heuristics
Suspicious: word_count ≤ 3 OR (word_count < 10 AND has_superlatives)
Genuine: word_count ≥ 5 OR has_domain_words

# Feature Extraction
TF-IDF with unigrams → Logistic Regression → Suspicion Scores

# SHAP Analysis
Select genuine reviews → Compute SHAP values → Find negative contributors
```

---

## Results Analysis

### Dataset Statistics
- **Total Books**: 150 matching criteria
- **Your Book**: "Tom Jones - The Life" (ASIN: 000810445X)
- **Total Reviews**: 14 for the identified book
- **Labeled Reviews**: 10 (6 suspicious, 4 genuine)

### SHAP Findings
Top words indicating genuine reviews:
1. **good** - Common in detailed feedback
2. **have** - Used in personal experience descriptions
3. **if** - Conditional statements in nuanced reviews

These words contrast with generic superlatives found in suspicious reviews.

---

## Project Structure

```
.
├── solver.py           # Main solution script
├── reflection.md       # Detailed approach documentation
├── README.md          # This file
├── books.csv          # Book dataset (input)
├── reviews.csv        # Reviews dataset (input)
└── flags.txt          # Generated flags (output)
```

---

## How It Works

### solver.py Workflow

```
1. Compute HASH_ID from student ID
   ↓
2. Filter books by rating criteria
   ↓
3. Find review containing HASH_ID
   ↓
4. Match book ASIN → Extract FLAG1
   ↓
5. Identify fake review → FLAG2
   ↓
6. Load all reviews for the book
   ↓
7. Engineer features (word count, superlatives, domain words)
   ↓
8. Label reviews (suspicious vs genuine)
   ↓
9. Train ML classifier (TF-IDF + Logistic Regression)
   ↓
10. Compute suspicion scores
   ↓
11. Select genuine reviews (exclude fake)
   ↓
12. Run SHAP analysis
   ↓
13. Find top 3 words reducing suspicion
   ↓
14. Compute FLAG3
   ↓
15. Save all flags to flags.txt
```

---

## Testing

To verify the solution:

```bash
# Run the solver
python solver.py

# Check output
cat flags.txt

# Should see:
# FLAG1 = 542523382ad4d0e10d6eacd47356e5decab43c238cc29c1d60f3bb18ebc0fbb4
# FLAG2 = FLAG2{F853BFAD}
# FLAG3 = FLAG3{5412dec5fa}
```






