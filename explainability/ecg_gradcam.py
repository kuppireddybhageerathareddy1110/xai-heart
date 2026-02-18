import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load model
model = tf.keras.models.load_model("models/ecg_cnn.h5", compile=False)

# load data
X = np.load("data/mitbih_X.npy")
X = (X - X.min())/(X.max()-X.min())

sample = X[100][np.newaxis, ..., np.newaxis]
sample = tf.cast(sample, tf.float32)

# find last Conv1D layer
last_conv = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv1D):
        last_conv = layer
last_conv_name = last_conv.name
print("Using layer:", last_conv_name)

# create feature extractor model
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=last_conv.output
)

# classifier model (everything after conv)
classifier_input = tf.keras.Input(shape=last_conv.output.shape[1:])
x = classifier_input
take = False
for layer in model.layers:
    if layer.name == last_conv_name:
        take = True
        continue
    if take:
        x = layer(x)

classifier_model = tf.keras.Model(classifier_input, x)

# compute gradcam
with tf.GradientTape() as tape:
    conv_output = feature_extractor(sample)
    tape.watch(conv_output)

    preds = classifier_model(conv_output)
    class_idx = tf.argmax(preds[0])
    loss = preds[:, class_idx]

grads = tape.gradient(loss, conv_output)

# channel importance
weights = tf.reduce_mean(grads, axis=1)

cam = tf.reduce_sum(weights * conv_output, axis=-1)[0]
cam = tf.maximum(cam, 0)
cam = cam / (tf.reduce_max(cam) + 1e-8)
cam = cam.numpy()

# resize to signal length
cam = np.interp(np.linspace(0, len(cam), 240), np.arange(len(cam)), cam)

# plot
plt.figure(figsize=(12,4))
plt.plot(sample.numpy()[0,:,0], label="ECG Signal")
plt.plot(cam, label="Model Attention", linewidth=2)
plt.legend()
plt.title("Grad-CAM Explanation (ECG)")
plt.show()
