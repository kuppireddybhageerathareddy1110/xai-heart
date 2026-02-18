import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading processed clinical dataset...")

# =========================
# LOAD PREPROCESSED DATA
# =========================
X = pd.read_csv("data/clinical_processed.csv")
y = np.load("data/clinical_target.npy")

print("Dataset:", X.shape)

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Clinical Accuracy:", acc)

# =========================
# SAVE MODEL + FEATURES
# =========================
model.save_model("models/clinical_xgb.json")
joblib.dump(X.columns.tolist(), "models/clinical_features.pkl")

print("Model saved ✔")

# =========================
# SHAP (stable version)
# =========================
print("Computing SHAP explanations...")

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])  # faster

# global importance
shap.summary_plot(shap_values, X_test[:100])

# local explanation
shap.plots.waterfall(shap_values[0])
