# 🤖 Gesture Recognition for Smart-TV Control Using Deep Learning

This project aims to implement a deep learning-based gesture recognition system that enables users to control a Smart-TV using five predefined hand gestures — offering a hands-free, remote-free experience.

## 📌 Problem Statement

Design and train a model that can classify video sequences of hand gestures into five categories to control a smart-TV. The system should work in real-time and be generalizable across different lighting and backgrounds.

---

## 🧪 Experiments Overview

Over **14 iterations**, we experimented with various architectures:

- **Conv3D** models with different layer depths and parameters
- **Conv2D + GRU** and **Conv2D + Dense** hybrids
- **ConvLSTM2D** models for temporal sequence learning
- Final attempt: **Conv3D + ConvLSTM2D** hybrid

Along the way, we integrated:
- `EarlyStopping`, `ReduceLROnPlateau` callbacks
- Batch Normalization, Dropout for stability
- Custom generators and preprocessing enhancements

### 🧠 Learnings:
- Conv3D models can easily overfit on small datasets
- Memory limitations affect batch size and model depth
- Adding temporal models like GRU/LSTM did not always improve validation accuracy
- Image augmentation and normalization were critical for generalization

---

## 🛠️ Getting Started

### Requirements
- Python 3.7+
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Open `model_training.ipynb` in Jupyter Notebook or Colab.
2. Execute each cell to preprocess the data and train the model.
3. Evaluate the model accuracy and loss using the plots.

---

## 📈 Results

| Model Variant        | Accuracy | Validation Loss |
|----------------------|----------|------------------|
| Conv3D (baseline)    | 1.00     | Overfitting      |
| Conv2D + GRU         | 0.6989   | 1.9708           |
| Conv2D + Dense       | 0.8139   | 1.3384           |
| ConvLSTM (various)   | ~0.56    | ~1.4–2.0         |
| Final Hybrid Model   | 0.4062   | 1.2165           |

---

## 📌 Future Improvements

- Implement data augmentation and adaptive learning schedules
- Try lightweight architectures like MobileNet3D or TCNs
- Use larger and more diverse datasets for better generalization
- Deploy model with real-time camera input on embedded devices

