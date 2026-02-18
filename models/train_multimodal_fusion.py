import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ShuffleSplit

# ---------------------------------------------------
# Load model outputs
# ---------------------------------------------------
y_raw = np.load("data/mitbih_y.npy")[:549]

# Binary abnormal detection
y = np.array([0 if str(x) == 'N' else 1 for x in y_raw])

ptb_feat = np.load("data/ptbxl_features.npy")[:549]
clinical = np.load("data/clinical_probs.npy")[:549]
rhythm = np.load("data/mitbih_probs.npy")[:549]

X = np.column_stack([clinical, rhythm, ptb_feat])

print("Fusion input:", X.shape)
print("Abnormal cases:", np.sum(y), "/", len(y))

# ---------------------------------------------------
# Repeated subsampling training (stable rare learning)
# ---------------------------------------------------
splitter = ShuffleSplit(n_splits=15, test_size=0.25, random_state=42)

models = []
aucs = []

for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # heavy penalty for missing patient
    model = LogisticRegression(
        max_iter=2000,
        class_weight={0:1, 1:60},
        solver="liblinear"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)

    print(f"Fold {fold+1:02d} AUC:", round(auc,3))

    models.append(model)
    aucs.append(auc)

print("\nMean validation AUC:", round(np.mean(aucs),3))

# ---------------------------------------------------
# Save ensemble
# ---------------------------------------------------
joblib.dump(models, "models/multimodal_heart_ai.pkl")
print("Saved ensemble model (15 learners)")
