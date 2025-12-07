# 🧠 NeuroScan AI: End-to-End Brain Tumor Detection

A deep learning medical diagnostic tool built to detect brain tumors from MRI scans. This project implements a full end-to-End pipeline: form data preprocessing and CNN model training to a user-friendly Streamlit web interface with built-in safety guardrails.

![Demo App Screenshot](put_your_screenshot_here.png)

## 🚀 Key Features

* **Deep Learning Core:** Uses a custom Convolutional Neural Network (CNN) trained on thousands of MRI scans.
* **Real-time Analysis:** Instant inference using TensorFlow.
* **Safety Guardrails:**
    * **Confidence Thresholds:** Rejects low-confidence predictions to prevent false positives.
    * **RGB Standardization:** Automatically handles 4-channel (RGBA) or Grayscale inputs to prevent tensor shape errors.
* **Interactive UI:** Built with Streamlit for a clean, medical-grade dashboard.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **ML Framework:** TensorFlow / Keras
* **Interface:** Streamlit
* **Image Processing:** NumPy, Pillow (PIL)

## 🧠 Model Architecture

The model uses a Sequential CNN architecture designed for pattern recognition in medical imaging:
1.  **Conv2D Layers:** To detect edges, textures, and tumor shapes.
2.  **MaxPooling:** To downsample and reduce computational load.
3.  **Dropout (0.5):** To prevent overfitting and encourage generalization.
4.  **Sigmoid Output:** For binary classification (Tumor vs. Healthy).

## 💻 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/neuroscan-ai.git](https://github.com/yourusername/neuroscan-ai.git)
    cd neuroscan-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run neuro_app.py
    ```

## 🛡️ Engineering Decisions & Guardrails

During development, I encountered the "Closed World Assumption" problem—where the AI would confidently classify a human face as a "Healthy Brain" simply because it didn't see a tumor.

To solve this, I implemented a **Confidence Logic Gate**:
```python
if confidence < 0.80:
    return "UNKNOWN", confidence