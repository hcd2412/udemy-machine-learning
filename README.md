# Udemy Machine Learning — End-to-End Applied ML (Python → Embedded C)

This repository contains **hands-on machine learning implementations** based on the *Udemy Machine Learning A–Z* course, extended with **production-grade structure**, **evaluation**, and **embedded deployment**.

The goal is not just to learn ML algorithms, but to demonstrate **end-to-end ML system design**, from data → model → metrics → deployment on constrained hardware.

---

## 📌 What makes this repo different

Most ML course repos stop at notebooks.  
This repo goes further:

- ✅ Config-driven training pipelines (YAML)
- ✅ Reproducible experiments
- ✅ Proper train/test evaluation
- ✅ Artifact persistence (models + metrics)
- ✅ **Automatic export to Embedded C**
- ✅ Ready for real-world / edge deployment discussions

---

## 🧠 Implemented Models

### 1️⃣ Simple Linear Regression (Salary Prediction)

- Python training pipeline
- Evaluation (MAE, RMSE, R²)
- Auto-generated **C header** with model coefficients
- Embedded inference in C

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
- Embedded C implementation

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
deploy/embedded # Embedded C inference examples
```

---

## 🔁 Python → Embedded Deployment Flow

1. Train model using a config-driven Python pipeline  
2. Evaluate using standard regression metrics  
3. Persist trained model and metrics artifacts  
4. Automatically export learned coefficients to a C header  
5. Run deterministic inference on embedded / edge targets  

This ensures a **single source of truth** between Python experimentation  
and deployed embedded inference.

---

## 🔧 Tech Stack

- Python, NumPy, Pandas, scikit-learn
- YAML-based configuration
- Joblib for model persistence
- C (embedded-friendly inference)
- Git with clean branching strategy

---

## 🎯 Why this matters

This repository demonstrates:

- ML **fundamentals**
- Software engineering discipline
- Awareness of **deployment constraints**
- Ability to explain ML beyond notebooks

It is designed to support applications for:

- **Senior / Applied Data Scientist**
- **ML Engineer**
- **AI / Analytics roles (UAE, GCC, global)**

---

## 🚀 Next Steps (in progress)

- Polynomial Regression
- Regularization (Ridge / Lasso)
- Model selection & bias–variance tradeoff
- More embedded-friendly ML patterns
