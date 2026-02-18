import numpy as np
import joblib
import shap
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, recall_score,
    precision_score, confusion_matrix, precision_recall_curve,
    balanced_accuracy_score, brier_score_loss, auc
)

from scipy.spatial.distance import cosine

print("\n==== MULTIMODAL AI VALIDATION ====\n")

# ---------------------------------------------------
# STEP 1 — Binary labels
# ---------------------------------------------------
y_raw = np.load("data/mitbih_y.npy")[:549]
y_true = np.array([0 if str(x) == 'N' else 1 for x in y_raw])

# ---------------------------------------------------
# STEP 2 — Load model outputs
# ---------------------------------------------------
clinical_probs = np.load("data/clinical_probs.npy")[:549]
rhythm_probs = np.load("data/mitbih_probs.npy")[:549]
ptbxl_features = np.load("data/ptbxl_features.npy")[:549]

fusion_input = np.column_stack([clinical_probs, rhythm_probs, ptbxl_features])

# Ensemble prediction
fusion_models = joblib.load("models/multimodal_heart_ai.pkl")
fusion_probs = np.mean([m.predict_proba(fusion_input)[:,1] for m in fusion_models], axis=0)

print("Samples:", len(y_true))
print("Abnormal ratio:", np.mean(y_true))

# ---------------------------------------------------
# STEP 3 — Proper threshold selection (split)
# ---------------------------------------------------
split = int(len(y_true)*0.6)

train_y, test_y = y_true[:split], y_true[split:]
train_p, test_p = fusion_probs[:split], fusion_probs[split:]

precision, recall, thresholds = precision_recall_curve(train_y, train_p)

beta = 2
f2 = (1+beta**2)*(precision[:-1]*recall[:-1])/((beta**2*precision[:-1])+recall[:-1]+1e-6)
best_threshold = thresholds[np.argmax(f2)]

print("\nChosen clinical threshold:", round(best_threshold,3))

fusion_pred = (fusion_probs >= best_threshold).astype(int)
clinical_pred = (clinical_probs >= 0.5).astype(int)
rhythm_pred = (rhythm_probs >= 0.5).astype(int)

# ---------------------------------------------------
# STEP 4 — Metrics
# ---------------------------------------------------
def evaluate(name, y, pred, prob):

    acc = accuracy_score(y, pred)
    bal = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    sens = recall_score(y, pred)
    spec = recall_score(y, pred, pos_label=0)
    prec = precision_score(y, pred)
    aucroc = roc_auc_score(y, prob)

    pr_prec, pr_rec, _ = precision_recall_curve(y, prob)
    pr_auc = auc(pr_rec, pr_prec)

    brier = brier_score_loss(y, prob)

    return [name, acc, bal, f1, aucroc, pr_auc, sens, spec, prec, brier]

results = []
results.append(evaluate("Clinical", y_true, clinical_pred, clinical_probs))
results.append(evaluate("Rhythm", y_true, rhythm_pred, rhythm_probs))
results.append(evaluate("Fusion", y_true, fusion_pred, fusion_probs))

df = pd.DataFrame(results, columns=[
    "Model","Accuracy","BalancedAcc","F1","ROC_AUC","PR_AUC",
    "Sensitivity","Specificity","Precision","Brier"
])

print("\n=== PERFORMANCE COMPARISON ===")
print(df.to_string(index=False))

# ---------------------------------------------------
# STEP 5 — Bootstrap confidence interval (AUC)
# ---------------------------------------------------
def bootstrap_auc(y, prob, n=200):
    scores=[]
    rng=np.random.default_rng(42)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))<2:
            continue
        scores.append(roc_auc_score(y[idx],prob[idx]))
    return np.percentile(scores,[2.5,50,97.5])

ci = bootstrap_auc(y_true,fusion_probs)
print("\nFusion ROC-AUC 95% CI:", np.round(ci,3))

# ---------------------------------------------------
# STEP 6 — Confusion matrix
# ---------------------------------------------------
tn, fp, fn, tp = confusion_matrix(y_true, fusion_pred).ravel()
print("\nCONFUSION MATRIX")
print(f"TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

# ---------------------------------------------------
# STEP 7 — SHAP validation
# ---------------------------------------------------
print("\n=== SHAP FEATURE VALIDATION ===")

xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/clinical_xgb.json")

clinical_df = pd.read_csv("data/clinical_processed.csv").iloc[:549]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(clinical_df)

importance = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": clinical_df.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print(shap_df.head(10).to_string(index=False))

# ---------------------------------------------------
# STEP 8 — Explanation stability
# ---------------------------------------------------
def stability_score(values):
    sims=[1-cosine(values[i],values[i+1]) for i in range(20)]
    return np.mean(sims)

print("\nSHAP Stability:", round(stability_score(shap_values),3))

print("\n==== VALIDATION COMPLETE ====\n")
