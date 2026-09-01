# Cross-Validation, Stability & Error Analysis

**Sprint:** Sprint 2, Day 3 · **Priority:** High
**Author:** _fill in your name_ **Date:** _fill in submission date_

> **Note on data:** This analysis runs on the same synthetic churn dataset and `random_state = 42`
> used in Day 1, recreated inside `Day3_Cross_Validation_Analysis.ipynb` so the notebook is
> self-contained. Replace the data-loading cell with your real Sprint 1 dataset and every
> number in this report will update accordingly when you re-run the notebook.

## 1. Objective

Determine whether candidate-model performance is stable across different data subsets, rather
than relying on a single validation result, and analyze *why* models succeed or fail — not just
how well they score on average.

## 2. Cross-Validation Strategy

**Selected: Stratified K-Fold, K = 5.**

| Consideration | This project |
|---|---|
| Problem type | Binary classification |
| Class balance | Imbalanced (~27% positive class) |
| Grouped observations? | No — one row = one independent customer |
| Temporal ordering? | No time-series structure in the Sprint 1 features |

Stratification is required because plain K-Fold could produce folds with very different
positive-class rates given the ~27/73 imbalance, which would make fold-to-fold comparisons
unreliable. K = 5 is used as the technically-justified default: with ~3,200 train/validation
rows this gives ~640 rows per validation fold — enough to estimate F1/ROC-AUC with reasonably
low noise, while keeping the per-fold training cost reasonable for the slower candidates (SVM,
Gradient Boosting). Group K-Fold and Time-Series Split were considered and ruled out: there is
no grouping key and no temporal ordering in this dataset.

## 3. Models Evaluated

The baseline plus all five candidates carried forward from Day 1 (all five beat the baseline on
mean F1 in the Day 1 single-split comparison):

- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM (RBF kernel)
- k-Nearest Neighbors

All models share the identical preprocessing pipeline, train/validation data, random seed, and
now the identical 5-fold Stratified CV splitter — so any performance difference below is
attributable to the algorithm, not to inconsistent setup. The final test set (800 rows, 20%,
stratified) remains untouched, exactly as locked away in Day 1.

## 4. Required Evidence — Cross-Validation Results

*(Full machine-readable version: `Cross_Validation_Results.csv`)*

| Model | Mean CV Score | Standard Deviation | Min Score | Max Score | Baseline Difference | Decision |
|---|---|---|---|---|---|---|
| SVM (RBF kernel) | 0.9118 | 0.0084 | 0.8969 | 0.9209 | +0.1108 | Candidate for tuning |
| Random Forest | 0.8899 | 0.0107 | 0.8765 | 0.9063 | +0.0889 | Reject — stability concern |
| Gradient Boosting | 0.8894 | 0.0083 | 0.8757 | 0.9000 | +0.0884 | Reject — stability concern |
| k-Nearest Neighbors | 0.8788 | 0.0140 | 0.8589 | 0.8963 | +0.0778 | Candidate for tuning |
| Decision Tree | 0.8292 | 0.0150 | 0.8136 | 0.8571 | +0.0282 | Reject — stability concern |
| Logistic Regression (baseline) | 0.8010 | 0.0147 | 0.7831 | 0.8226 | 0.0000 | Baseline (reference only) |

*(Metric = F1-score, positive class = churn, mean/std/min/max across 5 stratified folds.)*

![Mean CV F1 vs. baseline, with ±1 std error bars](images/plot_1.png){width=95%}

![Per-fold score distribution — stability check](images/plot_2.png){width=95%}

**Reading the results:** every candidate still beats the baseline on mean F1, consistent with
Day 1. But the mean alone is not sufficient to pick a model — see Section 5.

## 5. Stability Diagnostics

Three specific failure patterns were checked for each model, not just the aggregate mean:

| Model | Mean Val F1 | Mean Train F1 | Train-Val Gap | Fold Range (Max−Min) | Overfitting Flag | High-Variance Flag | Fold-Inconsistency Flag |
|---|---|---|---|---|---|---|---|
| SVM (RBF kernel) | 0.9118 | 0.9395 | 0.0277 | 0.0240 | No | No | No |
| Random Forest | 0.8899 | **1.0000** | **0.1101** | 0.0298 | **Yes** | No | No |
| Gradient Boosting | 0.8894 | 0.9414 | 0.0520 | 0.0243 | **Yes** | No | No |
| k-Nearest Neighbors | 0.8788 | 0.8935 | 0.0147 | 0.0374 | No | No | No |
| Decision Tree | 0.8292 | **1.0000** | **0.1708** | 0.0436 | **Yes** | No | No |
| Logistic Regression (baseline) | 0.8010 | 0.8093 | 0.0083 | 0.0395 | No | No | No |

*(Flag thresholds: High Variance = std > 0.02; Overfitting = train-val gap > 0.05;
Fold-Inconsistency = fold range > 0.06 — applied consistently to every model.)*

**Findings:**

- **Random Forest and Decision Tree memorize the training folds** — both reach a **perfect
  1.000 training F1**, while validation F1 sits far lower (0.890 and 0.829 respectively). This
  is a textbook overfitting signature: excellent on data the model has seen, materially worse on
  held-out folds.
- **Gradient Boosting** shows the same pattern at a smaller scale (train F1 = 0.941 vs.
  validation F1 = 0.889, a 0.052 gap) — still large enough to flag given the threshold applied
  uniformly across all models.
- **SVM (RBF) and k-Nearest Neighbors have the smallest train-validation gaps** (0.028 and
  0.015) among the non-baseline models, meaning their cross-validation scores are the most
  trustworthy estimate of true generalization.
- **No model is flagged for high fold-to-fold variance or fold inconsistency** at the applied
  thresholds — the main stability failure in this run is overfitting (train-val gap), not
  fold-to-fold instability.
- Note the baseline's own fold range (0.0395) is close to Decision Tree's — a reminder that a
  moderate fold range by itself isn't disqualifying; it is the *combination* of a large train-val
  gap with tree-based model family behavior that drives the rejection decisions here.

## 6. Classification Error Analysis

All confusion matrices and precision/recall figures below use **out-of-fold predictions**
(`cross_val_predict` with the same Stratified 5-Fold splitter) — every prediction was made by a
model instance that never saw that row during training.

![Confusion matrices — SVM, Random Forest, Gradient Boosting](images/plot_3.png){width=100%}

| Model | True Neg. | False Pos. | False Neg. | True Pos. | False Positive Rate | False Negative Rate |
|---|---|---|---|---|---|---|
| SVM (RBF kernel) | 2245 | 77 | 78 | 800 | 3.32% | 8.88% |
| Random Forest | 2287 | 35 | 146 | 732 | 1.51% | 16.63% |
| Gradient Boosting | 2266 | 56 | 130 | 748 | 2.41% | 14.81% |

**False positive analysis** (predicted churn, customer actually stayed): Random Forest has the
fewest false positives (35) and the lowest false-positive rate (1.51%) of the three — it is the
most conservative about flagging churn, which limits wasted retention spend but, as shown next,
comes at a cost.

**False negative analysis** (predicted no churn, customer actually left): SVM has by far the
fewest false negatives (78, 8.88% FN rate) versus Random Forest (146, 16.63%) and Gradient
Boosting (130, 14.81%). Since a missed churner is the costlier error for a retention use case
(lost revenue vs. a wasted discount), **SVM's error profile is the better fit for this business
problem**, not just its higher F1.

**Precision/recall trade-off:**

![Precision-recall curves — out-of-fold probabilities](images/plot_4.png){width=75%}

SVM's curve sits above Random Forest and Gradient Boosting across most of the recall range,
meaning that at any chosen decision threshold, SVM can generally achieve a better
precision/recall combination than the two tree ensembles on this dataset. Per-class detail
(precision/recall/F1) for the baseline and the two strongest candidates:

```
--- Logistic Regression (baseline) ---
              precision    recall  f1-score   support
    No Churn       0.95      0.89      0.92      2322
       Churn       0.74      0.87      0.80       878
    accuracy                           0.88      3200

--- SVM (RBF kernel) ---
              precision    recall  f1-score   support
    No Churn       0.97      0.97      0.97      2322
       Churn       0.91      0.91      0.91       878
    accuracy                           0.95      3200

--- Random Forest ---
              precision    recall  f1-score   support
    No Churn       0.94      0.98      0.96      2322
       Churn       0.95      0.83      0.89       878
    accuracy                           0.94      3200
```

Random Forest's churn-class precision (0.95) is higher than SVM's (0.91), but its churn-class
recall (0.83) is notably lower than SVM's (0.91) — consistent with the false-negative analysis
above: Random Forest is precise but misses more actual churners.

## 7. Top 2 Models Selected for Hyperparameter Tuning

**Selected: SVM (RBF kernel) and k-Nearest Neighbors.**

The decision combines mean CV score, stability (variance / fold range), and the
train-validation gap — not the mean score in isolation:

1. **SVM (RBF kernel)** — highest mean CV F1 (0.9118), smallest train-validation gap among all
   non-baseline models (0.028, no overfitting flag), lowest false-negative rate of the models
   inspected in error analysis, and the strongest precision-recall curve. Clear top pick.
2. **k-Nearest Neighbors** — second among the *stability-clean* models: no overfitting flag
   (train-val gap of only 0.015, the smallest of any candidate), no variance or fold-consistency
   flag. Its mean CV F1 (0.8788) is lower than Random Forest's and Gradient Boosting's, but
   those two are excluded on stability grounds (see below) before ranking by score.

## 8. Why the Remaining Models Are Rejected

- **Random Forest (mean F1 0.8899, 2nd highest raw score) — rejected.** Flagged for
  overfitting: 1.000 training F1 vs. 0.890 validation F1 (gap = 0.110), the largest gap of any
  model. Its cross-validation score cannot be fully trusted as a generalization estimate when
  the model perfectly memorizes every training fold.
- **Gradient Boosting (mean F1 0.8894, 3rd highest raw score) — rejected.** Same failure mode
  at a smaller scale: train F1 0.941 vs. validation F1 0.889 (gap = 0.052), above the applied
  overfitting threshold. Also carries a materially higher false-negative rate (14.81%) than SVM
  (8.88%) on the costlier error type for this business problem.
- **Decision Tree (mean F1 0.8292, lowest of the ensemble/kernel models) — rejected.** The most
  extreme overfitting signature observed (train F1 = 1.000, val F1 = 0.829, gap = 0.171) combined
  with the lowest mean validation score among the tree/ensemble/kernel family — dominated by
  both criteria.
- **Logistic Regression (baseline) — not a rejection, but not carried forward either.** Retained
  purely as the reference point every candidate must beat (per the Day 1 protocol); Day 1 and
  Day 3 both confirm every candidate beats it on the primary metric, so there is nothing left to
  gain by re-tuning the baseline itself at this stage.

*(This reasoning is generated directly from `decision_df` in the notebook, not hard-coded — if
you re-run the notebook against your real Sprint 1 dataset and the flags land differently, this
section's model names should be updated to match.)*

## 9. Acceptance Criteria

- [x] Cross-validation correctly implemented — Stratified 5-Fold, consistent with Day 1's protocol, test set never touched
- [x] Stability evaluated using mean **and** variation — Section 4 (std/min/max) + Section 5 (train-val gap, fold range)
- [x] Errors analyzed rather than only overall scores — Section 6 (confusion matrices, FP/FN rates, precision-recall trade-off)
- [x] Top 2 models selected using evidence — Section 7, derived from the combined score + stability table
- [x] Rejection decisions justified — Section 8, tied to specific numeric flags per model

## 10. Portal Submission Checklist

- [ ] `Day3_Cross_Validation_Analysis.ipynb`
- [ ] `Cross_Validation_Results.csv`
- [ ] `Day3_Model_Analysis.md` / `.pdf` (this document)
- [ ] Updated project repository link: ______________________________
