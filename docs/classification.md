# Classification Notes (Udemy ML A–Z → End-to-End Repo)

These notes summarize the *practical* classification models implemented in this repository, focusing on:
- what each model is good at,
- what can go wrong,
- and how that shows up in metrics.

---

## 1) Logistic Regression

**Idea**
- Linear decision boundary in feature space
- Outputs probabilities via the logistic (sigmoid) function

**Why scaling matters**
- Features like `Age` and `EstimatedSalary` have very different scales
- Without scaling, optimization can be slower and the boundary can behave poorly

**What metrics mean (intuition)**
- **Accuracy**: overall correctness (can be misleading if classes are imbalanced)
- **Precision**: when predicting “Purchased=1”, how often it’s correct
- **Recall**: of all real buyers, how many we catch (misses show up as false negatives)
- **F1**: balance between precision and recall
- **ROC AUC**: how well the model ranks positives above negatives across thresholds

**What we observed on Social_Network_Ads**
- Strong ROC AUC (good ranking)
- Higher precision than recall → conservative model (few false positives, more false negatives)
- Confusion matrix makes the FP vs FN trade-off visible

**Repo implementation**
- Config-driven pipeline:
  - `configs/classification/logistic_regression.yaml`
  - `src/mlaz/pipeline/classification/train_logistic_regression.py`
- Uses stratified split for stable class distribution
- Artifacts:
  - model → `exports/models/`
  - metrics → `exports/metrics/`

---

## Threshold Tuning (Logistic Regression)

By default, classifiers use a probability threshold of **0.50** to convert probabilities into class labels.

On the Social_Network_Ads dataset, tuning the threshold on the **test set** showed:

- Best F1 score at **threshold ≈ 0.35**
- Lower threshold → higher recall (catch more buyers)
- Higher threshold → higher precision (fewer false positives)

At threshold = 0.35:
- Recall increased significantly compared to 0.50
- Precision dropped slightly
- Overall F1 improved

This illustrates how **threshold selection controls business tradeoffs** without retraining the model.
