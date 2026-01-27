# Simple Linear Regression (Salary) — Embedded Inference

This folder contains a minimal embedded-C inference implementation for the
Simple Linear Regression model trained in Python on the Udemy Salary dataset.

## Model
Salary = intercept + coef * YearsExperience

Coefficients are stored in:
- `model_coeffs.h`

Inference function:
- `inference.c` → `slr_predict_salary(double years_experience)`

## Notes
- This example uses `double` for simplicity. On real MCU targets, you may switch
  to `float` or fixed-point depending on hardware/FPU support.
- The coefficients are generated from the Python training pipeline under `src/`.
