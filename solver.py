import hashlib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import numpy as np

ID = "STU032"
full_hash = hashlib.sha256(ID.encode()).hexdigest()
HASH_ID = full_hash[:8].upper()

print("Student ID:", ID)
print("SHA256 Hash:", full_hash)
print("HASH_ID:", HASH_ID)

books_df = pd.read_csv("books.csv")

filtered_books = books_df[
    (books_df["rating_number"] == 1234) &
    (books_df["average_rating"] == 5.0)
]

print("Books matching criteria:", len(filtered_books))
if len(filtered_books) == 0:
    raise SystemExit("No matching books found")

reviews_df = pd.read_csv("reviews.csv")

matching_reviews = reviews_df[
    reviews_df["text"].str.contains(HASH_ID, case=False, na=False)
]

print("reviews containing HASH_ID:", len(matching_reviews))
if len(matching_reviews) == 0:
    raise SystemExit("No reviews contain HASH_ID")

candidate_asins = set(matching_reviews["asin"].dropna().unique()) | \
                  set(matching_reviews["parent_asin"].dropna().unique())

print("Candidate ASINs:", candidate_asins)

book_asins = set(filtered_books["parent_asin"]) & candidate_asins
if len(book_asins) == 0:
    raise SystemExit("Could not match book to review ASINs")

book_asin = list(book_asins)[0]
book = filtered_books[filtered_books["parent_asin"] == book_asin].iloc[0]

print("ASIN:", book_asin)
print("Title:", book["title"])

title_prefix = "".join(c for c in book["title"] if c != " ")[:8]
print("Title prefix:", title_prefix)

FLAG1 = hashlib.sha256(title_prefix.encode()).hexdigest()
print("FLAG1:", FLAG1)

FLAG2 = f"FLAG2{{{HASH_ID}}}"
print("FLAG2:", FLAG2)

book_reviews = reviews_df[
    (reviews_df["asin"] == book_asin) |
    (reviews_df["parent_asin"] == book_asin)
].copy()

needed_cols = ["text", "rating"]
for col in needed_cols:
    if col not in book_reviews.columns:
        raise SystemExit(f"Column '{col}' missing in reviews.csv")

print("Total reviews for this book:", len(book_reviews))

superlatives = ["best", "amazing", "awesome", "must-read",
                "perfect", "incredible", "great", "excellent", "wonderful"]
book_words = ["characters", "plot", "narrative", "writing",
              "pacing", "worldbuilding", "prose", "story", "book", 
              "read", "author", "chapter", "good", "jones", "life"]

short_threshold = 3
long_threshold = 5

def label_review(text, rating):
    if not isinstance(text, str):
        return None

    text_lower = text.lower()
    word_count = len(text_lower.split())

    if rating != 5:
        return None

    has_super = any(w in text_lower for w in superlatives)
    has_book_word = any(w in text_lower for w in book_words)

    if word_count <= short_threshold or ((word_count < 10) and has_super):
        return 1
    
    if word_count >= long_threshold or has_book_word:
        return 0

    return None

labels = []
clean_texts = []
ratings = []

for _, row in book_reviews.iterrows():
    text = row["text"]
    rating = row["rating"]
    lab = label_review(text, rating)
    if lab is not None:
        labels.append(lab)
        clean_texts.append(text)
        ratings.append(rating)

if len(labels) < 5:
    raise SystemExit(f"Not enough labeled reviews for training, got {len(labels)}")

labeled_df = pd.DataFrame({
    "text": clean_texts,
    "rating": ratings,
    "label": labels,
})

print("Labeled reviews:", len(labeled_df))
print("Suspicious (1):", (labeled_df["label"] == 1).sum())
print("Genuine (0):", (labeled_df["label"] == 0).sum())

vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_all = vectorizer.fit_transform(labeled_df["text"])
y_all = labeled_df["label"].values

X_all_dense = X_all.toarray()

clf = LogisticRegression(max_iter=1000)
clf.fit(X_all_dense, y_all)

proba = clf.predict_proba(X_all_dense)[:, 1]
labeled_df["suspicion_score"] = proba

is_genuine = labeled_df["label"] == 0
no_hash_id = ~labeled_df["text"].str.contains(HASH_ID, case=False, na=False)

genuine_df = labeled_df[is_genuine & no_hash_id].copy()

if genuine_df.empty:
    raise SystemExit("No genuine reviews left after excluding HASH_ID review")

genuine_df = genuine_df.sort_values("suspicion_score", ascending=True)
n_samples = min(50, len(genuine_df))
genuine_sample = genuine_df.head(n_samples)

print("Genuine reviews used for SHAP:", len(genuine_sample))

X_genuine = vectorizer.transform(genuine_sample["text"]).toarray()

explainer = shap.Explainer(clf, X_all_dense)
shap_values = explainer(X_genuine)

if hasattr(shap_values, "values"):
    sv = shap_values.values
else:
    sv = shap_values

mean_shap = sv.mean(axis=0)

feature_names = vectorizer.get_feature_names_out()
sorted_idx = np.argsort(mean_shap)

top_k = 3
top_indices = sorted_idx[:top_k]
top_words = [feature_names[i] for i in top_indices]

print("Top 3 words reducing suspicion:", top_words)

clean_words = [w.replace(" ", "").lower() for w in top_words]
numeric_id = "".join(ch for ch in ID if ch.isdigit())

base_str = "".join(clean_words) + numeric_id
flag3_hash = hashlib.sha256(base_str.encode()).hexdigest()
flag3_code = flag3_hash[:10]

FLAG3 = f"FLAG3{{{flag3_code}}}"
print("FLAG3:", FLAG3)

with open("flags.txt", "w") as f:
    f.write(f"FLAG1 = {FLAG1}\n")
    f.write(f"FLAG2 = {FLAG2}\n")
    f.write(f"FLAG3 = {FLAG3}\n")
