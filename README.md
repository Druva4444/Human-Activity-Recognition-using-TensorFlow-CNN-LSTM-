# Human Activity Recognition using ConvLSTM on UCF50 Dataset

This project implements **Human Activity Recognition (HAR)** using **Convolutional LSTM (ConvLSTM) networks** trained on the [UCF50 dataset](https://www.crcv.ucf.edu/data/UCF50.php).  
The model leverages spatiotemporal features extracted from video frames to classify human actions such as *HighJump* and *PoleVault*.

---

## 📌 Features
- Downloads and extracts the **UCF50 dataset** automatically.
- Preprocesses videos into fixed-length frame sequences.
- Implements a **ConvLSTM-based deep learning model** for activity recognition.
- Supports **early stopping** to avoid overfitting.
- Includes visualization of:
  - Random sample frames from the dataset.
  - Training/validation accuracy and loss curves.

---

## 📂 Project Structure
```
├── main.ipynb / main.py      # Training and evaluation code
├── UCF50/                    # Dataset (downloaded automatically)
├── README.md                 # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/human-activity-recognition-convlstm.git
cd human-activity-recognition-convlstm
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
The script automatically downloads and extracts the **UCF50 dataset**:
```python
!wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar
!unrar x UCF50.rar
```

---

## 📊 Dataset
- **UCF50** dataset contains **50 different human action categories**.  
- This project uses only a subset:
  - `HighJump`
  - `PoleVault`
- Each video is processed into **20 frames** of size `244x244`.

---

## 🧠 Model Architecture
The network is a **ConvLSTM** architecture with multiple layers:

- ConvLSTM2D layers with increasing filter sizes (4 → 16).
- MaxPooling3D layers for temporal-spatial downsampling.
- Dropout layers for regularization.
- Flatten + Dense softmax for classification.

---

## 🚀 Training
To train the ConvLSTM model:
```python
convlstm_model = create_convlstm_model()

convlstm_model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy']
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True
)

convlstm_model.fit(
    x=features_train,
    y=labels_train,
    epochs=50,
    batch_size=4,
    shuffle=True,
    validation_split=0.2,
    callbacks=[early_stopping_callback]
)
```

---

## 📈 Visualization
Training curves can be plotted using:

```python
plot_metric(convlstm_model_training_history, 'accuracy', 'val_accuracy', 'Model Accuracy')
plot_metric(convlstm_model_training_history, 'loss', 'val_loss', 'Model Loss')
```

---

## 📌 Requirements
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- MoviePy
- scikit-learn

Install all requirements:
```bash
pip install tensorflow opencv-python numpy matplotlib moviepy scikit-learn
```

---

## 🔮 Future Work
- Extend to all **50 UCF50 classes**.
- Experiment with **3D CNN + LSTM** hybrids.
- Deploy trained models as a **real-time activity recognition system**.

---

## 📜 License
This project is released under the MIT License.

---

## 🙌 Acknowledgments
- [UCF50 Dataset](https://www.crcv.ucf.edu/data/UCF50.php)  
- TensorFlow & Keras team  
- OpenCV community  
