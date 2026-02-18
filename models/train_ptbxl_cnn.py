import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.load("data/ptbxl_X.npy")
y = np.load("data/ptbxl_y.npy")

# normalize
X = (X - X.mean()) / (X.std() + 1e-8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1000,12)),

    tf.keras.layers.Conv1D(32,7,activation="relu"),
    tf.keras.layers.MaxPool1D(2),

    tf.keras.layers.Conv1D(64,5,activation="relu"),
    tf.keras.layers.MaxPool1D(2),

    tf.keras.layers.Conv1D(128,5,activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train,y_train,epochs=10,validation_split=0.2)

model.save("models/ptbxl_ecg_model.keras")
