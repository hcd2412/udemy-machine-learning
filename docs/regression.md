# Regression Notes (Udemy ML A–Z → End-to-End Repo)

These notes summarize the *practical* regression models implemented in this repository, focusing on:
- when each model works well,
- when it fails,
- and how that shows up in metrics/plots.

This is intentionally **applied** (not textbook-heavy), because the goal is to build end-to-end ML + deployment intuition.

---

## 1) Simple Linear Regression (SLR)

**Model**
- One feature → one target:
  - `y = intercept + coef * x`

**When it works well**
- Relationship is roughly linear
- Small datasets (low model complexity)
- You need interpretability

**Common failure modes**
- Nonlinear relationship → systematic error (high bias)
- Outliers can dominate the fit

**Repo implementation**
- Python training + metrics
- Auto-export to C header (single coefficient)
- Embedded C inference example

---

## 2) Multiple Linear Regression (MLR)

**Model**
- Multiple features:
  - `y = intercept + Σ (coef[i] * x[i])`

**When it works well**
- Additive linear effects explain the target
- You want interpretable feature impact
- You can encode categorical variables (one-hot)

**Common failure modes**
- Multicollinearity can make coefficients unstable
- High-cardinality categorical variables can explode feature count
- Still linear → can miss nonlinear effects

**Repo implementation**
- One-hot encoding for categorical features
- Auto-export to C header (array of coefficients + feature names)
- Embedded C inference example

---

## 3) Polynomial Regression

**Idea**
- Still linear regression under the hood, but we expand the feature space:
  - `x → [x, x², x³, ...]`

**Bias–variance intuition**
- Higher degree → lower bias but higher variance
- Easy to overfit small datasets

**What to look for**
- Very high train R² with significantly worse test metrics → overfitting warning

**Repo implementation**
- Config-driven degree selection
- Prints/records metrics that help compare under/overfitting

---

## 4) Regularization: Ridge vs Lasso

### Ridge Regression (L2)
**Objective**
- `min ||y − Xβ||² + λ||β||²`

**What it does**
- Shrinks coefficients smoothly
- Helps when features are correlated
- Reduces variance (often improves generalization)

### Lasso Regression (L1)
**Objective**
- `min ||y − Xβ||² + λ||β||₁`

**What it does**
- Can push some coefficients exactly to zero → implicit feature selection
- Useful when you suspect many weak/irrelevant features

**Practical notes**
- With tiny datasets, regularization may not “save” a bad feature set
- Regularization helps most when there are many features relative to samples

---

## 5) Support Vector Regression (SVR)

**Why scaling matters**
- SVR distance-based behavior depends on feature scale
- Always scale features (and often target, depending on formulation)

**Kernel intuition**
- RBF kernel can fit nonlinear patterns, but hyperparameters dominate behavior:
  - `C` controls penalty for errors
  - `epsilon` sets an error-insensitive tube
  - `gamma` controls how “wavy” the RBF curve can become

**Why SVR can look poor on Position_Salaries**
- Dataset is extremely small (`n = 10`)
- Salary jumps are steep/non-smooth (nearly exponential)
- Depending on `epsilon` and `gamma`, SVR can underfit heavily and produce a flat-ish curve
- Metrics like R² can go negative when the model performs worse than predicting the mean

**What the plot should show**
- Actual points vs a smooth SVR curve on top
- If the curve misses the steep jump at high levels, that’s expected under certain hyperparameters

---

## How to use this doc

When you add a new regression model:
1. Add a short section here (5–10 bullets max)
2. Capture what worked / what failed
3. Point to the repo pipeline + config filenames
4. (Optional) mention an important plot or metric pattern you observed
