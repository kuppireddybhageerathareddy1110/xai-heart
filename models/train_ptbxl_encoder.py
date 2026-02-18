import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.load("data/ptbxl_X.npy")

# normalize per record
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True)+1e-8)

X_train, X_test = train_test_split(X, test_size=0.2)

# -------- Encoder --------
inp = tf.keras.Input(shape=(1000,12))

x = tf.keras.layers.Conv1D(32,7,padding="same",activation="relu")(inp)
x = tf.keras.layers.MaxPool1D(2)(x)

x = tf.keras.layers.Conv1D(64,5,padding="same",activation="relu")(x)
x = tf.keras.layers.MaxPool1D(2)(x)

x = tf.keras.layers.Conv1D(128,5,padding="same",activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)

latent = tf.keras.layers.Dense(64,name="embedding")(x)

# -------- Decoder --------
x = tf.keras.layers.Dense(250*128,activation="relu")(latent)
x = tf.keras.layers.Reshape((250,128))(x)

x = tf.keras.layers.UpSampling1D(2)(x)
x = tf.keras.layers.Conv1D(64,5,padding="same",activation="relu")(x)

x = tf.keras.layers.UpSampling1D(2)(x)
out = tf.keras.layers.Conv1D(12,7,padding="same")(x)

autoencoder = tf.keras.Model(inp,out)
encoder = tf.keras.Model(inp,latent)

autoencoder.compile(optimizer="adam",loss="mse")

autoencoder.fit(X_train,X_train,epochs=15,batch_size=32,validation_data=(X_test,X_test))

encoder.save("models/ptbxl_encoder.keras")
print("PTB-XL encoder saved")
