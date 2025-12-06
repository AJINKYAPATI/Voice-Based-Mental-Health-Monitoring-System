# 🎤 Speech Emotion Recognition Using LSTM

An advanced **Speech Emotion Recognition (SER)** system built using **LSTM Neural Networks** in TensorFlow/Keras, integrated with a **Streamlit Web App** for interactive predictions. This project leverages deep learning to classify human emotions from audio speech signals.

![Speech Emotion Recognition](https://github.com/user-attachments/assets/4b1378b3-04c9-4afd-9f81-8b075b46e8b0)


---

## 📌 Project Overview

This project processes audio signals, extracts meaningful features, and classifies them into one of the following **7 emotion classes**:

* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😊
* Neutral 😐
* Sad 😢
* Surprise 😲

The system includes:

* **Data Preprocessing & Feature Extraction**
* **Augmentation Techniques** for robust training
* **LSTM-based Deep Learning Model**
* **Model Evaluation (Accuracy \~97%)**
* **Streamlit UI for Real-Time Emotion Prediction**

---

## 📂 Dataset

The model is trained on the **TESS (Toronto Emotional Speech Set)** dataset:

* **Language**: English
* **Speakers**: 2 (Female)
* **Emotions**: 7 (as listed above)
* **Audio Format**: WAV, 16-bit PCM, 24kHz

Dataset Source: [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## ⚙️ Features Used

The model uses **essential audio features** extracted via **Librosa**:

* **Zero Crossing Rate**
* **Chroma STFT**
* **MFCC (13 coefficients)**
* **RMS Energy**

These features capture pitch, energy, timbre, and harmonic structure of speech.

---

## 🔍 Model Architecture

* **Input Shape**: (27, 1)
* **Layers**:

  * LSTM Layers (with Dropout)
  * Dense Layers
  * Softmax Output
* **Loss Function**: Categorical Crossentropy
* **Optimizer**: Adam
* **Performance**: **97% Accuracy** on test data

---

## 📊 Model Performance

```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       518
           1       0.89      0.98      0.94       480
           2       1.00      1.00      1.00       505
           3       0.98      0.93      0.96       502
           4       1.00      1.00      1.00       500
           5       1.00      0.92      0.96       470
           6       0.95      0.97      0.96       525

    accuracy                           0.97      3500
   macro avg       0.97      0.97      0.97      3500
weighted avg       0.97      0.97      0.97      3500
```

---

## 🖥️ Streamlit Web Application

The project includes a **Streamlit app** with the following features:

✅ **Upload Audio File** (WAV)
✅ **Real-Time Emotion Prediction**
✅ **Probability Visualization** using:

* Bar Charts
* Radar Charts
* Donut Charts
  ✅ **Professional UI with Custom CSS**

### 🔗 UI Screenshots
<img width="881" height="807" alt="donutChart" src="https://github.com/user-attachments/assets/1161f8e4-bedd-4fbf-9ff7-aabce135745f" />
<img width="1408" height="882" alt="barChart" src="https://github.com/user-attachments/assets/f7e5bac6-f3f7-466b-a4dd-4dc91c62179f" />
<img width="510" height="767" alt="confidenceScore" src="https://github.com/user-attachments/assets/927651ef-87f1-4a9d-ab13-f82aa7cc2a92" />
<img width="847" height="726" alt="ranking" src="https://github.com/user-attachments/assets/1c7bf417-4e02-4df2-8069-161d54ac31d8" />


---

## 📁 Project Structure

```
SpeechEmotionRecognition/
│
├── app.py                  # Streamlit Application
├── model/
│   ├── best_lstm_model.keras
│   ├── scaler.joblib
│   ├── encoder.joblib
│
├── Utilities/
│   ├── utils.py            # Feature extraction functions
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── notebooks/
    └── EmotionRecognitionFromSpeech.ipynb (HTML & IPYNB)
```

---

## 🚀 How to Run

### ✅ **Locally**

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/SpeechEmotionRecognition.git
   cd SpeechEmotionRecognition
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:

   ```bash
   streamlit run app.py
   ```


---

## 🛠️ Technologies Used

* **Python 3.10+**
* **TensorFlow / Keras** (LSTM Model)
* **Librosa** (Audio Feature Extraction)
* **Streamlit** (Web UI)
* **Joblib** (Model & Scaler Persistence)

---

## 📚 Future Improvements

* Add **multi-language support**
* Implement **real-time audio recording in Streamlit**
* Deploy on **AWS/GCP** with GPU acceleration

---

## 👨‍💻 Author

**Muhammad Ahsan Ikhlaq**
[LinkedIn Profile](https://www.linkedin.com/in/muhammad-ahsan-ikhlaq/)

---

## 📜 License

This project is licensed under the MIT License.
