import numpy as np
import pandas as pd

# clinical safe ranges (cardiology prior knowledge)
MEDICAL_LIMITS = {
    "age": (20, 90),
    "chol": (120, 300),
    "oldpeak": (0.0, 6.0),
    "thalch": (60, 210)
}

def generate_counterfactual(model, patient_df, target=0.25, steps=200):

    original = patient_df.copy()
    best = None
    best_dist = 1e9

    for _ in range(steps):

        candidate = original.copy()

        # randomly adjust numeric medical features
        for col in MEDICAL_LIMITS:
            if col in candidate.columns:
                lo, hi = MEDICAL_LIMITS[col]
                candidate[col] = np.clip(
                    candidate[col] + np.random.normal(0, (hi-lo)*0.05),
                    lo, hi
                )

        prob = model.predict_proba(candidate)[0,1]

        if prob < target:
            dist = np.linalg.norm(candidate.values - original.values)

            if dist < best_dist:
                best_dist = dist
                best = candidate.copy()

    return best
