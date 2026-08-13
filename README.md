# Quora Duplicate Question Pairs

Quora gets millions of questions a day, and a lot of them ask the same thing in different words. This project tackles that problem directly: given two questions, predict whether they're duplicates.

## The dataset

Quora's [Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) — 400k+ pairs of questions, each labeled as duplicate or not. I worked with a random sample of 30,000 pairs for modeling (the full set was used for the initial EDA).

## What I did

Started simple, then built up. First, I ran a plain Bag-of-Words model directly on the raw questions just to see how far basic word overlap gets you — around 74-75% accuracy, which is a decent floor.

From there I engineered a set of similarity features for each question pair: things like shared word/stopword ratios, length differences, longest common substring, and fuzzy-match scores (via RapidFuzz). Combined these with two different text representations — TF-IDF and a custom-trained Word2Vec model — and tested both against Random Forest and XGBoost.

Word2Vec + engineered features + XGBoost came out on top, so I tuned it further with RandomizedSearchCV.

## Results

| Model | TF-IDF | Word2Vec |
|---|---|---|
| Random Forest | 78.3% | 79.7% |
| XGBoost | 79.3% | 82.4% |
| XGBoost (tuned) | — | **82.5%** |

Final model — tuned XGBoost + Word2Vec:

- **Accuracy:** 82.5%
- **Precision:** 77.6%
- **Recall:** 73.6%
- **F1:** 75.5%
- **AUC:** 90.9%

**Takeaways:** engineered features gave a real lift over raw Bag-of-Words. Word2Vec beat TF-IDF as the underlying representation, and XGBoost beat Random Forest every time. Tuning only added ~0.1%, so the model was pretty close to its ceiling given the feature set and sample size.

## Stack

Python, Pandas/NumPy, Scikit-learn, XGBoost, Gensim (Word2Vec), NLTK, RapidFuzz, BeautifulSoup, Matplotlib/Seaborn/Plotly.

## Running it

```bash
pip install numpy pandas seaborn matplotlib scikit-learn xgboost gensim rapidfuzz distance nltk beautifulsoup4 plotly
```

Grab `train.csv` from the [Kaggle dataset](https://www.kaggle.com/c/quora-question-pairs), point the notebook at it, and run top to bottom.

## What's next

Training on the full 400k+ dataset, trying pretrained/transformer embeddings (Sentence-BERT), testing Logistic Regression or LightGBM as extra baselines, and eventually wrapping the final model in a small API or app.
