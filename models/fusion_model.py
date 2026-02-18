import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ======================================================
# 1) MIT-BIH RHYTHM MODEL
# ======================================================
ecg_model = tf.keras.models.load_model("models/ecg_cnn.h5", compile=False)

X_ecg = np.load("data/mitbih_X.npy")
y_ecg = np.load("data/mitbih_y.npy")

X_ecg = (X_ecg - X_ecg.min())/(X_ecg.max()-X_ecg.min())
X_ecg = X_ecg[..., np.newaxis]

ecg_probs = ecg_model.predict(X_ecg, verbose=0)
mit_risk = 1 - ecg_probs[:,0]

print("MIT risk:", mit_risk.shape)

# ======================================================
# 2) CLINICAL MODEL
# ======================================================
feature_cols = joblib.load("models/clinical_features.pkl")

X_clin = pd.read_csv("data/clinical_processed.csv")
X_clin = X_clin.reindex(columns=feature_cols, fill_value=0)

y_clin = np.load("data/clinical_target.npy")

clin_model = xgb.XGBClassifier()
clin_model.load_model("models/clinical_xgb.json")

clin_probs = clin_model.predict_proba(X_clin)[:,1]

print("Clinical:", clin_probs.shape)

# ======================================================
# 3) PTB-XL MORPHOLOGY FEATURES
# ======================================================
ptb_features = np.load("data/ptbxl_features.npy")
print("PTBXL:", ptb_features.shape)

# ======================================================
# 4) ALIGN DATASET SIZES
# ======================================================
n = len(y_clin)

mit_feature = np.interp(
    np.linspace(0, len(mit_risk)-1, n),
    np.arange(len(mit_risk)),
    mit_risk
)

ptb_feature = ptb_features[
    np.linspace(0, len(ptb_features)-1, n).astype(int)
]

# ======================================================
# 5) FINAL MULTIMODAL MATRIX
# ======================================================
X_fusion = np.column_stack([clin_probs, mit_feature, ptb_feature])
y_fusion = y_clin[:n]

print("Fusion shape:", X_fusion.shape)

# ======================================================
# 6) TRAIN META MODEL
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_fusion, y_fusion, test_size=0.2, stratify=y_fusion, random_state=42
)

meta_model = LogisticRegression(max_iter=2000)
meta_model.fit(X_train, y_train)

preds = meta_model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("\nFINAL 3-MODALITY ACCURACY:", acc)

joblib.dump(meta_model, "models/final_fusion.pkl")
