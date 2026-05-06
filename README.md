# Handwritten Digit Recognizer

A CNN trained on MNIST that classifies handwritten digits (0-9) via a Streamlit web interface.

\---

## Project Structure

```
mnist\_digit\_recognition/
├── train.py              # Train and save the model
├── app.py                # Streamlit web GUI
├── requirements.txt      # Dependencies
└── model/                # Auto-created by train.py
    ├── mnist\_cnn.keras
    ├── training\_history.png
    └── results.txt
```

\---

## Model Architecture

```
Input (28x28x1)
Conv2D(32) x2 + BN + ReLU -> MaxPool -> Dropout(0.25)
Conv2D(64) x2 + BN + ReLU -> MaxPool -> Dropout(0.25)
Conv2D(128)    + BN + ReLU           -> Dropout(0.25)
Flatten -> Dense(256) + BN + ReLU -> Dropout(0.5)
Dense(10, softmax)
```

Trained with Adam, sparse categorical cross-entropy, early stopping, and ReduceLROnPlateau. Data augmentation (rotation, translation, zoom) is applied during training.

\---

## Setup

```bash
git clone https://github.com/<your-username>/mnist-digit-recognizer.git
cd mnist-digit-recognizer
Python -m venv .venv 
.venv\\Scripts\\activate
pip install -r requirements.txt
```

\---

## Usage

Train the model:

```bash
python train.py
```

Launch the app:

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a digit image, and get an instant prediction.

\---

## Performance

|Metric|Value|
|-|-|
|Accuracy|\~99.4%|
|Loss|<0.03|

\---

## Tech Stack

Python, TensorFlow/Keras, Pillow, NumPy, Matplotlib, Streamlit

\---

## Repository

https://github.com/Haufz/mnist-digit-recognizer

\---

## License

MIT

