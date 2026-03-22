# Recommendation-system
A full end-to-end **product recommendation system** built on the Amazon Beauty dataset using **Singular Value Decomposition (SVD)** — a model-based collaborative filtering technique. The system learns latent user and product preferences from historical ratings, then generates personalised top-N product recommendations for every user.
# 🛍️ Amazon Product Recommendation System
### Collaborative Filtering using SVD (Matrix Factorization)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.11-8CAAE6?logo=scipy)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A full end-to-end **product recommendation system** built on the Amazon Beauty dataset using **Singular Value Decomposition (SVD)** — a model-based collaborative filtering technique. The system learns latent user and product preferences from historical ratings, then generates personalised top-N product recommendations for every user.

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Source | Amazon Beauty Ratings Dataset |
| File | `ratings_Beauty.csv` |
| Raw interactions | 2,023,070 |
| Raw users | 1,210,271 |
| Raw products | 249,274 |
| Rating scale | 1.0 – 5.0 |
| Columns | `UserId`, `ProductId`, `Rating`, `Timestamp` |

After filtering (≥ 5 ratings per user and product):

| Property | Value |
|----------|-------|
| Interactions | 198,502 |
| Users | 22,363 |
| Products | 12,101 |
| Sparsity | 99.93% |

---

## 🏗️ Project Structure

```
amazon-recommendation-system/
│
├── ratings_Beauty.csv          # Raw dataset
├── svd_recommender.py          # Full pipeline script
├── README.md                   # Project documentation
│
└── notebooks/
    └── recommendation_system.ipynb   # Colab notebook
```

---

## ⚙️ Pipeline

```
Raw Data (2M rows)
      ↓
Data Cleaning
  • Convert Unix timestamp → datetime
  • Drop duplicates
  • Filter: ≥5 ratings per user & product (iterative)
      ↓
Feature Engineering
  • Encode UserId & ProductId → integer indices
  • Temporal weight column (newer = higher weight)
      ↓
Train / Test Split  (80 / 20)
      ↓
Build Sparse CSR Matrix (time-weighted)
      ↓
SVD Model Training  (k = 50 latent factors)
      ↓
Evaluation  →  RMSE | MAE | Precision@10 | Recall@10
      ↓
Generate Recommendations  →  Top-N per user
      ↓
Batch Output  →  223,620 recommendation rows
```

---

## 🤖 Model — SVD (Truncated)

The core model uses **truncated Singular Value Decomposition** on a mean-centred user-item rating matrix.

**Key steps:**
1. Build a sparse `(n_users × n_products)` matrix from training ratings
2. Mean-centre each user's ratings to remove rating-scale bias
3. Apply truncated SVD: decompose into `U × Σ × Vt` with `k=50` latent factors
4. Reconstruct the full prediction matrix and clip to [1, 5]
5. Recommend top-N unseen products per user based on predicted scores

**Why SVD?**
- Handles explicit ratings (1–5 stars) natively
- Fast inference — predictions come from a single matrix lookup
- Cosine similarity in latent space enables "similar product" discovery
- Top-5 latent factors explain ~16% of variance (healthy for a sparse beauty dataset)

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **RMSE** | 1.1831 |
| **MAE** | 0.9245 |
| **Precision@10** | 0.0034 |
| **Recall@10** | 0.0159 |
| **F1@10** | 0.0056 |

> ⚠️ Low Precision/Recall values are **expected** — they are a direct consequence of 99.93% matrix sparsity. With very few known positives per user in the test set, even strong models score low on ranking metrics. RMSE and MAE are the primary accuracy signals here.

---

## 🎯 Sample Output

**Top-10 recommendations for user `A2V5R832QCSOMX`:**

| Rank | ProductId | Predicted Score |
|------|-----------|----------------|
| 1 | B00639DLV2 | 4.402 |
| 2 | B00H2B2RLK | 4.362 |
| 3 | B00AHF1GTM | 4.294 |
| 4 | B002WTC37A | 4.275 |
| 5 | B000WYZ9Q4 | 4.249 |

**Products similar to `B004OHR1Q`** (cosine similarity in latent space):

```
1. B002M3KEOU
2. B00ANTDQL8
3. B00GMQDN64
4. B003C1V814
5. B004KRV8MS
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/amazon-recommendation-system.git
cd amazon-recommendation-system
```

### 2. Install dependencies
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### 3. Add the dataset
Place `ratings_Beauty.csv` in the root directory.  
Download from: [Amazon Product Reviews – Kaggle](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)

### 4. Run the pipeline
```bash
python svd_recommender.py
```

Or open in **Google Colab**:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🔧 Key Functions

```python
# Get top-10 recommendations for a user
get_recommendations("A2V5R832QCSOMX", top_n=10)

# Find similar products
similar_products("B004OHR1Q", top_n=5)

# Generate recommendations for all users
batch_recs = batch_recommend(top_n=10)
batch_recs.to_csv("recommendations.csv", index=False)
```

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.0 | Data loading & manipulation |
| `numpy` | ≥ 1.24 | Matrix operations |
| `scipy` | ≥ 1.11 | Sparse matrix + SVD |
| `scikit-learn` | ≥ 1.3 | Train/test split, RMSE |
| `matplotlib` | ≥ 3.7 | Visualisations |
| `seaborn` | ≥ 0.12 | Plot styling |

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning — grid search over `n_factors` (50, 100, 200)
- [ ] Regularised SVD++ using the `Surprise` library
- [ ] ALS model for better implicit feedback handling
- [ ] Temporal drift modelling using the `Timestamp` column
- [ ] Streamlit / Gradio web UI for live recommendations
- [ ] Docker containerisation for deployment

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- Dataset: [Amazon Product Reviews](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings) via Kaggle
- Algorithm: Koren, Y. (2008). *Factorization meets the neighborhood: a multifaceted collaborative filtering model.*
