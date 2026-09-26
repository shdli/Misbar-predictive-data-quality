# MISBAR
### Predictive Data Quality & Anomaly Detection for E-Commerce Data Warehouses

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2)

---

## Table of Contents

1. [Overview](#overview)
2. [What MISBAR Does](#what-misbar-does)
3. [Dataset](#dataset)
4. [Pipeline](#pipeline)
5. [Error Injection](#error-injection)
6. [Feature Engineering](#feature-engineering)
7. [Models](#models)
8. [Results](#results)
9. [Explainability with SHAP](#explainability-with-shap)
10. [Key Findings](#key-findings)
11. [Limitations & Future Work](#limitations--future-work)
12. [How to Run](#how-to-run)
13. [Tech Stack](#tech-stack)
14. [Contributors](#contributors)

---

## Overview

In e-commerce data warehouses, bad records such as negative prices, impossible dates, or payments that don't match the order total silently break reports, dashboards, and business decisions. Traditional rule-based checks only catch the errors someone thought to write a rule for.

**MISBAR** is a predictive data quality system. Instead of relying on fixed rules alone, it uses machine learning to learn what valid records look like, predict which records are likely faulty, and explain *why* each one was flagged, so data teams know exactly which column to fix.

---

## What MISBAR Does

- **Predicts** faulty records across price, freight, payment, weight, and date fields
- **Combines** two supervised models with an unsupervised Autoencoder into one ensemble score
- **Explains** every flagged record using SHAP, pointing to the columns responsible
- **Breaks down** performance per error type to show where detection is strong and where it struggles

---

## Dataset

We used the [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), keeping **6 of the 9 tables**:

| Table | Rows |
|---|---|
| Orders | 99,441 |
| Order Items | 112,650 |
| Payments | 103,886 |
| Products | 32,951 |
| Customers | 99,441 |
| Sellers | 3,095 |

After cleaning and merging at the order-item level, the final table contains **112,650 rows × 36 columns**.

---

## Pipeline

```
Raw CSVs → Exploration → Cleaning → Merge → EDA → Clean Baseline
        → Error Injection → Feature Engineering → Train/Val/Test Split
        → Random Forest + XGBoost + Autoencoder → Ensemble → SHAP Explanations
```

**Cleaning:** duplicate checks, conversion of date columns to `datetime`, median/zero imputation for missing product attributes, and aggregation of payments per order.

**EDA:** payment method distribution, top-selling product categories, order status distribution, and monthly order and payment trends.

**Clean baseline:** business rules removed 248 invalid rows (non-positive prices or payments, negative freight, out-of-order dates, delivered orders without a delivery date), and values above the 99.9th percentile were trimmed. The result is a trusted baseline of **112,075 rows**.

---

## Error Injection

To evaluate the models against known ground truth, we injected controlled errors into **14%** of the baseline (15,690 rows). Each faulty row contains exactly one error.

| Category | Description | Error Types | Rows |
|---|---|---|---|
| **Simple** | Obviously invalid values | `negative_price`, `zero_price`, `negative_freight`, `negative_payment`, `broken_date` | 5,603 |
| **Moderate** | Plausible but inflated values (×5 to ×15) | `price_outlier`, `freight_outlier`, `weight_outlier` | 4,483 |
| **Complex** | Each value looks valid alone, but columns contradict each other | `delivered_before_purchase`, `approved_before_purchase`, `delivered_no_date`, `payment_mismatch` | 5,604 |

---

## Feature Engineering

Complex errors are invisible when looking at a single column, so we created cross-column features:

- **Time intervals:** approval hours, delivery days, carrier days, estimated days, late days, shipping limit days, carrier-to-customer days
- **Consistency ratios:** freight ratio, expected order total, payment difference, payment ratio, price per gram, product volume
- **Completeness flags:** number of missing values, delivery status, presence of a delivery date
- **Purchase time:** hour, month, day of week

Categorical columns were one-hot or label encoded, producing a final feature matrix of **51 features**.

**Split:** 70% train / 15% validation / 15% test, stratified to keep the 14% error rate in every split. The validation set is used only for threshold tuning; the test set is used only for final evaluation.

---

## Models

### 1. Random Forest
200 trees with `class_weight='balanced'` to handle the class imbalance.

### 2. XGBoost
300 estimators, `max_depth=6`, `learning_rate=0.1`, with `scale_pos_weight ≈ 6.14` to handle the class imbalance.

### 3. Autoencoder (Unsupervised)
A neural network trained **only on normal rows** to reconstruct its input. Faulty rows produce a high reconstruction error.

- **Architecture:** 28 → 64 → 32 → **16** → 32 → 64 → 28
- **Input:** 28 standardized numeric features, clipped to [-10, 10]
- **Training:** 67,467 normal rows, 30 epochs, MSE loss
- **Threshold:** selected on the validation set by maximizing F1

> Using only numeric features was a deliberate choice: including the one-hot columns reduced F1 from about 0.86 to 0.61.

The Autoencoder never sees labels, which gives it the potential to catch error patterns the supervised models were never trained on.

### 4. Ensemble
Each model outputs a score between 0 and 1 (the Autoencoder error is normalized by its 99.9th percentile on validation). The final score is the **average of the three**, with the decision threshold (0.31) tuned on the validation set.

---

## Results

Evaluated on the held-out test set (16,812 rows):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.9941 | 0.9286 | 0.9602 |
| XGBoost | 0.9785 | 0.9669 | 0.9726 |
| Autoencoder | 0.8548 | 0.8076 | 0.8305 |
| **Ensemble (3 models)** | **0.9856** | **0.9613** | **0.9733** |

All models except the Autoencoder exceed the **0.90 target** on all three metrics. The ensemble achieves the highest F1 by balancing Random Forest's high precision with XGBoost's high recall.

> Autoencoder and ensemble results may vary slightly between runs due to TensorFlow randomness.

### Detection Rate per Error Type

| Error Type | Random Forest | XGBoost | Autoencoder |
|---|---|---|---|
| weight_outlier | 0.383 | 0.685 | 0.171 |
| freight_outlier | 0.928 | 0.987 | 0.970 |
| price_outlier | 0.938 | 0.981 | 0.803 |
| payment_mismatch | 0.995 | 1.000 | 1.000 |
| negative_price | 1.000 | 0.994 | 0.866 |
| negative_freight | 1.000 | 1.000 | 0.886 |
| delivered_before_purchase | 1.000 | 1.000 | 0.135 |
| All other types | 1.000 | 1.000 | 1.000 |

---

## Explainability with SHAP

Flagging a row is not enough for a data quality tool: the person fixing it needs to know *which column* is wrong.

- **SHAP TreeExplainer** on XGBoost shows each feature's contribution to the anomaly score, both globally (summary plot) and per record (waterfall plot)
- For the Autoencoder, the column with the highest reconstruction error is reported

**Examples from the test set:**

| Injected Error | Ensemble Score | Top SHAP Drivers |
|---|---|---|
| `zero_price` | 0.671 | `freight_ratio`, `price_per_gram` |
| `negative_payment` | 0.608 | `payment_value`, `payment_ratio` |
| `freight_outlier` | 0.671 | `freight_value`, `order_freight_total` |

In each case, the top drivers point directly to the column that was actually corrupted. This turns a generic *"this row is suspicious"* into an actionable *"this row is suspicious because the payment does not match the order total."*

---

## Key Findings

1. **XGBoost is the strongest single model**, with Random Forest close behind.
2. **The ensemble gives the best overall F1**, but only slightly above XGBoost, since the two tree models already agree on most rows.
3. **Engineered features matter more than raw columns.** Date intervals and payment ratios dominate feature importance; without them, complex errors would be undetectable.
4. **`weight_outlier` is the hardest error.** Real product weights already range from grams to tens of kilograms, so a ×5 weight still looks plausible.
5. **The Autoencoder is weaker overall but complementary**, since it learns what "normal" looks like without relying on labels. It struggles most with date-order errors such as `delivered_before_purchase`.

---

## Limitations & Future Work

- Errors were synthetically injected; real warehouse errors are likely messier and scores would be lower
- The Autoencoder could be improved with tuning (depth, bottleneck size, epochs)
- Weighted or stacked ensembles could replace simple averaging
- Deployment as a scheduled monitoring job with alerting

---

## How to Run

```bash
git clone https://github.com/shdli/Misbar-predictive-data-quality.git
cd Misbar-predictive-data-quality
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow shap
```

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Place the 6 CSV files in the project root
3. Open `MISBAR_Github.ipynb` in Visual Studio Code, Google Colab, or Jupyter Notebook
4. Run all cells **in order**

---

## Tech Stack

| Purpose | Tools |
|---|---|
| Data processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn, XGBoost |
| Deep learning | TensorFlow / Keras |
| Explainability | SHAP |
| Environment | Visual Studio Code, Google Colab, Jupyter Notebook |

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/shdli">
        <img src="https://github.com/shdli.png" width="100px;" alt="Shahad Alzamil"/><br/>
        <sub><b>Shahad Alzamil</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/LatifahMohmmead">
        <img src="https://github.com/LatifahMohmmead.png" width="100px;" alt="Latifah Altamimi"/><br/>
        <sub><b>Latifah Altamimi</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/THsaeed">
        <img src="https://github.com/THsaeed.png" width="100px;" alt="Thuraya Alshahrani"/><br/>
        <sub><b>Thuraya Alshahrani</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/abdullahbarakh">
        <img src="https://github.com/abdullahbarakh.png" width="100px;" alt="Abdullah Alsulami"/><br/>
        <sub><b>Abdullah Alsulami</b></sub>
      </a>
    </td>
  </tr>
</table>
