import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

model = xgb.XGBClassifier()
model.load_model("models/clinical_xgb.json")

X = pd.read_csv("data/clinical_processed.csv")

probs = model.predict_proba(X)[:,1]

np.save("data/clinical_probs.npy", probs)
print("Saved clinical probabilities")
