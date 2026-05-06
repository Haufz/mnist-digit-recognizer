import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

tf.random.set_seed(42)
np.random.seed(42)

MODEL_PATH = os.path.join("model", "mnist_cnn.keras")
PLOT_PATH  = os.path.join("model", "training_history.png")


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32") / 255.0
    x_test  = x_test [..., np.newaxis].astype("float32") / 255.0
    print(f"train: {len(x_train):,}  test: {len(x_test):,}")
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = models.Sequential([
        # block 1
        layers.Conv2D(32, 3, padding="same", input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # block 2
        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # block 3
        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.25),

        # head
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_augmenter():
    return tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.1),
    ])


def train(model, x_train, y_train, x_test, y_test):
    os.makedirs("model", exist_ok=True)

    aug = make_augmenter()

    def augment(x, y):
        return aug(x, training=True), y

    ds_train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(60000)
        .batch(128)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(256)
        .prefetch(tf.data.AUTOTUNE)
    )

    cbs = [
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                  monitor="val_accuracy", verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=1, min_lr=1e-6),
    ]

    return model.fit(ds_train, epochs=30, validation_data=ds_val, callbacks=cbs)


def evaluate(model, x_test, y_test, history):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy : {acc*100:.2f}%")
    print(f"Test loss     : {loss:.4f}\n")

    with open(os.path.join("model", "results.txt"), "w") as f:
        f.write(f"accuracy: {acc*100:.2f}%\nloss: {loss:.4f}\n")

    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history.history["accuracy"],     label="train")
    ax1.plot(epochs, history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history.history["loss"],     label="train")
    ax2.plot(epochs, history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Plot saved -> {PLOT_PATH}")


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.summary()
    history = train(model, x_train, y_train, x_test, y_test)
    evaluate(model, x_test, y_test, history)
    print(f"Model saved -> {MODEL_PATH}")
