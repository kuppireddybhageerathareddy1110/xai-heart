import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/ecg_cnn.h5")

X = np.load("data/mitbih_X.npy")
X = X[...,np.newaxis]

probs = model.predict(X, verbose=0)[:,1]

np.save("data/mitbih_probs.npy", probs)
print("Saved rhythm probabilities")
