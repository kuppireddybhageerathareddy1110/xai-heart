import numpy as np
import tensorflow as tf

encoder = tf.keras.models.load_model("models/ptbxl_encoder.keras")

X = np.load("data/ptbxl_X.npy")

# get morphology embeddings
features = encoder.predict(X, verbose=0)

print("PTBXL feature shape:", features.shape)
print("Example vector:", features[0][:10])

np.save("data/ptbxl_features.npy", features)

print("Saved PTBXL features:", features.shape)
