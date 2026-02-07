# ML Algorithms Reference

Curated, production-oriented reference implementations of core machine learning algorithms in Python.

This repository is designed as a long-term technical reference: clean implementations, consistent structure, and reproducible runs — intended to support applied forecasting, anomaly detection, and decision systems projects.

---

## 📌 Scope and Intent

This repository focuses on **foundational machine learning methods** and their correct, interpretable implementation.

It is **not intended as a domain-specific production system**.  
Instead, it serves as:
- a fundamentals refresher,
- a reference implementation library,
- and a technical bridge between theory and applied projects elsewhere in this profile.

---

## 🧠 Implemented Models

### 1️⃣ Simple Linear Regression (Salary Prediction)

- Python-based training pipeline
- Standard evaluation metrics (MAE, RMSE, R²)
- Explicit mapping from learned coefficients to deterministic inference logic
- Optional export of coefficients for embedded / constrained environments

📁 Location:
```
configs/regression/simple_linear.yaml
src/mlaz/pipeline/train_simple_linear.py
deploy/embedded/slr_salary/
```

---

### 2️⃣ Multiple Linear Regression (Startup Profit Prediction)

- Numerical + categorical features
- One-hot encoding
- Evaluation and model persistence
- Auto-generated **C header** for multi-feature inference
- Optional embedded inference example

📁 Location:
```
configs/regression/multiple_linear.yaml
src/mlaz/pipeline/train_multiple_linear.py
deploy/embedded/mlr_startups/
```

---

## 🏗️ Repository Structure

```
configs/        # Experiment configurations (YAML)
data/           # Raw and processed datasets
src/mlaz/       # Modular ML code (data, pipeline, evaluation, export)
exports/        # Training artifacts (models, metrics)
deploy/embedded # Optional embedded inference examples
```

---

## 🔁 Python → Embedded Mapping (Illustrative)

This repository includes **illustrative examples** showing how learned model
parameters can be exported from Python and reused in deterministic,
constrained environments.

These examples are intended to:
- clarify how ML models translate into simple numerical logic,
- highlight deployment considerations at a conceptual level,
- support understanding of ML beyond notebooks.

---

## 🔧 Tech Stack

- Python, NumPy, Pandas, scikit-learn
- YAML-based configuration
- Joblib for model persistence
- C (embedded-friendly inference examples)
- Git with clean branching strategy

---

## 🎯 How This Fits in the Broader Portfolio

This repository demonstrates:
- solid understanding of **machine learning fundamentals**
- disciplined, readable implementation style
- attention to evaluation and reproducibility
- ability to reason about ML beyond notebooks

Applied, real-world projects are maintained separately and focus on:
- fleet demand forecasting
- operational anomaly detection
- smart-city and transportation analytics

---

## 🚀 Next Steps (in progress)

- Polynomial Regression
- Regularization (Ridge / Lasso)
- Model selection & bias–variance tradeoff
- More embedded-friendly ML patterns
