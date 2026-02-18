import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

print("Loading UCI clinical dataset...")

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("heart_disease_uci.csv")

# =====================================================
# TARGET
# =====================================================
# num: 0 healthy | 1-4 disease
df["target"] = (df["num"] > 0).astype(int)

# remove unused identifiers
df = df.drop(columns=["num", "id"], errors="ignore")

# =====================================================
# CLEAN DATA
# =====================================================
# replace ? with NaN
df = df.replace("?", np.nan)

# convert numeric safely
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# fill missing values
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# =====================================================
# SPLIT FEATURES
# =====================================================
cat_cols = df.select_dtypes(include="object").columns.tolist()
if "target" in cat_cols:
    cat_cols.remove("target")

num_cols = [c for c in df.columns if c not in cat_cols + ["target"]]

# =====================================================
# ONE HOT ENCODING (sklearn ≥1.2 compatible)
# =====================================================
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[cat_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(cat_cols),
    index=df.index
)

# =====================================================
# SCALE NUMERIC FEATURES
# =====================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(df[num_cols])

scaled_df = pd.DataFrame(
    scaled,
    columns=num_cols,
    index=df.index
)

# =====================================================
# FINAL DATASET
# =====================================================
X = pd.concat([scaled_df, encoded_df], axis=1)
y = df["target"].astype(int)

print("Final dataset shape:", X.shape)

# =====================================================
# SAVE OUTPUTS
# =====================================================
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

X.to_csv("data/clinical_processed.csv", index=False)
np.save("data/clinical_target.npy", y.values)

joblib.dump(list(X.columns), "models/clinical_features.pkl")
joblib.dump(encoder, "models/clinical_encoder.pkl")
joblib.dump(scaler, "models/clinical_scaler.pkl")

print("Clinical dataset ready ✔")
