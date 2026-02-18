import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# load dataset
X = np.load("data/mitbih_X.npy")
y = np.load("data/mitbih_y.npy")

print("Loaded:", X.shape, y.shape)

# normalize signal (-1 to 1 range)
X = (X - X.min()) / (X.max() - X.min())

# CNN expects (samples, length, channels)
X = X[..., np.newaxis]

# encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# build CNN
model = models.Sequential([
    layers.Conv1D(32, 5, activation='relu', input_shape=(240,1)),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 3, activation='relu'),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation='relu'),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# train
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# save model
model.save("models/ecg_cnn.h5")
print("Model saved!")
