import os
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("model", "mnist_cnn.keras")

st.set_page_config(page_title="Digit Recognizer", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # stop early if the model file doesn't exist yet
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Run `python train.py` first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def otsu_threshold(arr: np.ndarray) -> int:
    """Compute optimal threshold that separates digit from background."""
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()
    best_thresh, best_var = 0, 0
    for t in range(1, 256):
        w0, w1 = hist[:t].sum(), hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0 = (hist[:t] * np.arange(t)).sum() / w0
        m1 = (hist[t:] * np.arange(t, 256)).sum() / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_thresh = var, t
    return best_thresh


def preprocess(img: Image.Image) -> np.ndarray:
    # convert to grayscale
    img = img.convert("L")

    # blur to suppress paper texture and gradient noise before thresholding
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    arr = np.array(img)

    light_background = arr.mean() > 127

    if light_background:
        # threshold on original to find dark digit pixels BEFORE inversion
        # avoids dark vignette edges being mistaken for digit after inversion
        thresh = otsu_threshold(arr)
        binary = (arr < thresh).astype(np.uint8)
    else:
        # dark background: digit is already the bright region
        thresh = otsu_threshold(arr)
        binary = (arr > thresh).astype(np.uint8)

    # find rows and cols that contain digit pixels
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if rows.any() and cols.any():
        # get tight bounding box around the digit
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # add padding proportional to digit size so it doesn't touch the edges
        pad = max(int((rmax - rmin) * 0.2), int((cmax - cmin) * 0.2), 4)
        rmin = max(rmin - pad, 0)
        rmax = min(rmax + pad, arr.shape[0] - 1)
        cmin = max(cmin - pad, 0)
        cmax = min(cmax + pad, arr.shape[1] - 1)

        arr = arr[rmin:rmax+1, cmin:cmax+1]

    # now invert so digit is white-on-black (MNIST convention)
    if light_background:
        arr = 255 - arr

    # binarize: force pure black/white to eliminate gradient background
    # without this, the murky gray background confuses the model
    clean_thresh = otsu_threshold(arr)
    arr = np.where(arr > clean_thresh, 255, 0).astype(np.uint8)

    # pad shorter axis so the crop is square (preserves aspect ratio)
    h, w = arr.shape
    if h != w:
        size = max(h, w)
        square = np.zeros((size, size), dtype=arr.dtype)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = arr
        arr = square

    # resize to 28x28 and normalise to [0, 1]
    img = Image.fromarray(arr).resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0

    # reshape to (1, 28, 28, 1) for model input
    return arr.reshape(1, 28, 28, 1)


def prob_chart(probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 2.5))

    # highlight the predicted class in blue
    colors = ["steelblue" if i == probs.argmax() else "lightgray" for i in range(10)]
    ax.bar(range(10), probs * 100, color=colors, width=0.6)
    ax.set_xticks(range(10))
    ax.set_xticklabels(map(str, range(10)))
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # annotate the top bar with its confidence value
    best = probs.argmax()
    ax.text(best, probs[best] * 100 + 2, f"{probs[best]*100:.1f}%",
            ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    model = load_model()

    st.title("Digit Recognizer")
    st.caption("Upload a handwritten digit image (0-9) for CNN classification.")
    st.divider()

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    img = Image.open(io.BytesIO(uploaded.read()))
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(img, width='stretch')

        # show exactly what gets fed into the model
        with st.expander("Preprocessed 28x28"):
            tensor = preprocess(img)
            thumb = Image.fromarray((tensor.squeeze() * 255).astype("uint8"), mode="L")
            st.image(thumb.resize((140, 140), Image.NEAREST))

    with col2:
        st.subheader("Prediction")
        tensor = preprocess(img)
        probs = model.predict(tensor, verbose=0)[0]  # shape (10,)
        digit = int(probs.argmax())
        conf  = float(probs.max()) * 100
        st.metric("Digit", digit)
        st.metric("Confidence", f"{conf:.1f}%")

    st.divider()
    st.subheader("Class Probabilities")
    st.pyplot(prob_chart(probs), width='stretch')

    with st.expander("Full probability table"):
        import pandas as pd
        df = pd.DataFrame({"digit": range(10), "confidence": [f"{p*100:.2f}%" for p in probs]})
        st.dataframe(df, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
