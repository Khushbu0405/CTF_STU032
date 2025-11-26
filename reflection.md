# CTF Challenge - Reflection and Approach

## Student ID: STU032

---

## Overview
This document outlines the methodology and approach used to solve the three-flag CTF challenge involving book reviews dataset analysis.

---

## FLAG1: Book Identification

### Objective
Find a specific book from the dataset and extract FLAG1 from its title.

### Approach
1. **Hash Computation**: Computed SHA256 hash of student ID "STU032" and extracted first 8 characters (uppercase): `F853BFAD`

2. **Book Filtering**: 
   - Filtered books.csv for books with:
     - `rating_number = 1234`
     - `average_rating = 5.0`
   - Found 150 books matching these criteria

3. **Review Scanning**:
   - Searched reviews.csv for reviews containing the HASH_ID `F853BFAD`
   - Found 1 review containing this hash
   - Extracted the book ASIN from this review: `000810445X`

4. **Book Identification**:
   - Matched the ASIN to the filtered books
   - Identified book: "Tom Jones - The Life"

5. **FLAG1 Computation**:
   - Extracted first 8 non-space characters from title: "TomJones"
   - Computed SHA256 hash of "TomJones"
   - **FLAG1**: `542523382ad4d0e10d6eacd47356e5decab43c238cc29c1d60f3bb18ebc0fbb4`

---

## FLAG2: Fake Review Identification

### Objective
Identify the fake review containing the planted hash.

### Approach
1. **Review Analysis**: The review containing our HASH_ID (`F853BFAD`) is the manipulated/fake review
2. **FLAG2 Format**: Simply the HASH_ID itself
3. **FLAG2**: `FLAG2{F853BFAD}`

---

## FLAG3: SHAP Analysis of Review Authenticity

### Objective
Use machine learning and SHAP (SHapley Additive exPlanations) to identify words that characterize genuine reviews.

### Approach

#### Step 1: Data Collection
- Filtered reviews to only those for our identified book (ASIN: 000810445X)
- Found 14 total reviews for this book
- Focused on 5-star reviews

#### Step 2: Feature Engineering
Calculated three key features for each review:
- **Word Count**: Length of the review text
- **Superlative Count**: Presence of generic praise words (best, amazing, awesome, excellent, wonderful, etc.)
- **Domain Word Count**: Presence of book-specific terminology (good, jones, life, book, read, story, etc.)

#### Step 3: Review Labeling
Created binary labels using heuristic rules:

**Suspicious Reviews** (Label = 1):
- Very short (≤3 words) OR
- Short (<10 words) AND contains superlatives

**Genuine Reviews** (Label = 0):
- Longer (≥5 words) OR
- Contains book-related domain words

Results:
- 10 reviews labeled total
- 6 suspicious reviews
- 4 genuine reviews

#### Step 4: Model Training
- **Vectorization**: Used TfidfVectorizer with unigrams to convert text to numerical features
- **Model**: Trained Logistic Regression classifier
- **Purpose**: Predict probability of review being suspicious (suspicion score)

#### Step 5: Suspicion Score Analysis
- Computed suspicion probability for each labeled review
- Lower scores indicate more genuine reviews
- Used these scores to identify most genuine reviews for SHAP analysis

#### Step 6: SHAP Analysis
- **Selected Sample**: 4 genuine reviews (those with label=0)
- **Excluded Fake Review**: Removed review containing HASH_ID to ensure clean analysis
- **SHAP Computation**: Used Linear SHAP explainer to understand feature contributions
- **Analysis Goal**: Find words that REDUCE suspicion (negative SHAP values)

#### Step 7: Top Words Identification
Identified top 3 words with most negative SHAP values (strongest indicators of genuine reviews):
1. "good" (SHAP: negative)
2. "have" (SHAP: negative)  
3. "if" (SHAP: negative)

These words push predictions away from "suspicious" towards "genuine"

#### Step 8: FLAG3 Computation
1. Concatenated top 3 words + numeric ID: `goodhaveif032`
2. Computed SHA256 hash
3. Took first 10 hex characters: `5412dec5fa`
4. **FLAG3**: `FLAG3{5412dec5fa}`

---

## Key Challenges and Solutions

### Challenge 1: Limited Data
**Problem**: Book had only 14 reviews, with very short text (average 5 words)

**Solution**: 
- Adjusted labeling thresholds from standard (20/40 words) to dataset-appropriate (3/5 words)
- Expanded domain word list to include common words found in the actual reviews
- Used all available labeled data instead of train/test split

### Challenge 2: Class Imbalance
**Problem**: Initial strict criteria resulted in too few or single-class labels

**Solution**:
- Made labeling rules more flexible (using OR conditions)
- Lowered thresholds to match actual data distribution
- Ensured both classes (suspicious and genuine) were represented

### Challenge 3: SHAP with Small Dataset
**Problem**: SHAP typically requires more samples for reliable results

**Solution**:
- Used Linear SHAP explainer (more stable for small data)
- Focused analysis on clearly genuine reviews (label=0)
- Removed fake review to avoid contaminating analysis

---

## Technical Stack

### Libraries Used
- **pandas**: Data loading and manipulation
- **scikit-learn**: 
  - TfidfVectorizer for text feature extraction
  - LogisticRegression for classification
- **shap**: SHAP values computation for model interpretability
- **hashlib**: SHA256 hash computation
- **numpy**: Numerical operations

### Files
- **Input**: `books.csv`, `reviews.csv`
- **Output**: `flags.txt`
- **Script**: `solver.py`

---

## Results Summary

| Flag | Value | Method |
|------|-------|--------|
| HASH_ID | F853BFAD | SHA256(STU032)[:8] |
| FLAG1 | 542523382ad4d0e10d6eacd47356e5decab43c238cc29c1d60f3bb18ebc0fbb4 | Book title analysis |
| FLAG2 | FLAG2{F853BFAD} | Fake review identification |
| FLAG3 | FLAG3{5412dec5fa} | SHAP + ML analysis |

---

## Lessons Learned

1. **Data Adaptation**: Machine learning models need thresholds adjusted to actual data distribution, not assumed values

2. **Feature Engineering**: Domain-specific features (superlatives, book words) were crucial for creating meaningful labels

3. **Explainability**: SHAP provided interpretable insights into what makes reviews appear genuine vs suspicious

4. **Heuristic Rules**: Simple rule-based labeling can be effective when ground truth labels are unavailable

5. **Negative SHAP Values**: Features with negative SHAP values for the suspicious class are the strongest indicators of genuine content

---

## Future Improvements

1. **More Data**: With more reviews, could use train/test split for validation
2. **Advanced Features**: Could add sentiment analysis, writing style metrics
3. **Ensemble Methods**: Could try Random Forest or other ensemble classifiers
4. **Cross-Validation**: With more data, could validate model stability
5. **SHAP Visualizations**: Could generate plots to better understand feature importance

---

## Conclusion

Successfully solved all three flags by combining:
- Data filtering and pattern matching (FLAG1, FLAG2)
- Machine learning classification (FLAG3)
- Model interpretability via SHAP (FLAG3)

The approach demonstrated practical application of ML/AI techniques for detecting anomalous or fake reviews in e-commerce datasets.
