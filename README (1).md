<div align="center">

# 🎙️ Voice-Based Mental Health Monitoring System

### *Speech Emotion Recognition Using Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

<br/>

> **Upload or record your voice — and let AI understand how you feel.**  
> A deep learning-powered web app that detects emotions from speech and provides personalized mental wellness recommendations.

<br/>

![App Banner](https://img.shields.io/badge/🧠_Powered_by-Deep_Learning-6366F1?style=flat-square) &nbsp;
![App Banner](https://img.shields.io/badge/🎵_Built_with-Streamlit-FF4B4B?style=flat-square) &nbsp;
![App Banner](https://img.shields.io/badge/🚀_Deployed_on-Streamlit_Cloud-00BFFF?style=flat-square)

</div>

---

## 📌 Table of Contents

- [✨ Features](#-features)
- [🎬 Demo](#-demo)
- [🧠 How It Works](#-how-it-works)
- [🏗️ System Architecture](#️-system-architecture)
- [📊 Emotion Classes](#-emotion-classes)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [📈 Results](#-results)
- [🔮 Future Enhancements](#-future-enhancements)
- [📚 Dataset](#-dataset)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎤 **Live Recording** | Record audio directly in the browser — no external tools needed |
| 📁 **Multi-Format Upload** | Supports WAV, MP3, OGG, M4A, FLAC, WEBA (up to 200MB) |
| 🤖 **Deep Learning Inference** | Real-time emotion classification with confidence scores |
| 📊 **Rich Visualizations** | Waveform, Mel Spectrogram, Bar Chart, Radar Chart, Donut Chart |
| 💡 **Wellness Suggestions** | Context-aware mental health tips based on detected emotion |
| 📋 **Detailed Analytics** | Emotion ranking table with statistical summary |
| ⚡ **Fast Processing** | Analysis completes in seconds |
| 🎨 **Clean UI** | Intuitive Streamlit interface with a modern design |

---

## 🎬 Demo

<div align="center">

### 📤 File Upload Flow
> Upload a `.wav` or audio file → View waveform & spectrogram → Click **Analyze Emotion** → Get results + wellness tips

### 🎙️ Live Recording Flow
> Open microphone recorder → Record your voice → Click **Analyze Emotion** → See your emotional state

</div>

**Sample Output:**

```
🎙️  Input  : OAF_base_disgust.wav  (119.1 KB)
🧠  Emotion : DISGUST
📊  Confidence : 100.0%
📈  Stats   : Highest: 100% | Avg: 14.3% | Std Dev: 35.0%
```

---

## 🧠 How It Works

The system follows a 7-step end-to-end pipeline:

```
  🎤 Audio Input
       │
       ▼
  📦 Preprocessing
  (Resample → Normalize → Load)
       │
       ▼
  📉 Visualization
  (Waveform + Mel Spectrogram)
       │
       ▼
  🔬 Feature Extraction
  (MFCCs · Spectral · Energy · Temporal)
       │
       ▼
  🤖 Deep Learning Inference
  (CNN / LSTM Model)
       │
       ▼
  📊 Results & Visualizations
  (Bar Chart · Radar · Donut · Table)
       │
       ▼
  💡 Wellness Recommendations
  (Emotion-based mental health tips)
```

### 🔍 Extracted Audio Features

The AI model analyzes the following acoustic features:

- 🎵 **Spectral Characteristics** — Spectral centroid, bandwidth, rolloff
- 🔊 **Voice Energy Patterns** — RMS energy, loudness dynamics
- 📊 **Frequency Distributions** — Mel-Frequency Cepstral Coefficients (MFCCs), Chroma
- ⏱️ **Temporal Dynamics** — Zero Crossing Rate, rhythm patterns
- 🌈 **Mel Spectrogram** — Full time-frequency representation (in dB)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APP                        │
├─────────────────┬───────────────────────────────────────────┤
│   SIDEBAR       │           MAIN CONTENT AREA               │
│ ─────────────── │ ─────────────────────────────────────────  │
│ • How It Works  │  ┌──────────────────────────────────────┐ │
│ • Supported     │  │  🎤 Audio Input Module               │ │
│   Formats       │  │  (Upload File / Live Microphone)     │ │
│ • Reset Button  │  └──────────────┬───────────────────────┘ │
│                 │                 │                          │
│                 │  ┌──────────────▼───────────────────────┐ │
│                 │  │  📉 Audio Visualization               │ │
│                 │  │  (Waveform + Mel Spectrogram)        │ │
│                 │  └──────────────┬───────────────────────┘ │
│                 │                 │                          │
│                 │  ┌──────────────▼───────────────────────┐ │
│                 │  │  🤖 Deep Learning Inference           │ │
│                 │  │  (Feature Extraction → Model)        │ │
│                 │  └──────────────┬───────────────────────┘ │
│                 │                 │                          │
│                 │  ┌──────────────▼───────────────────────┐ │
│                 │  │  📊 Results Dashboard                 │ │
│                 │  │  Predicted Emotion | Confidence Score │ │
│                 │  │  Bar Chart | Radar | Donut | Table   │ │
│                 │  └──────────────┬───────────────────────┘ │
│                 │                 │                          │
│                 │  ┌──────────────▼───────────────────────┐ │
│                 │  │  💡 Wellness Recommendations          │ │
│                 │  └──────────────────────────────────────┘ │
└─────────────────┴───────────────────────────────────────────┘
```

---

## 📊 Emotion Classes

The model classifies speech into **7 emotion categories**:

| Emoji | Emotion | Description |
|:---:|---|---|
| 😠 | **Angry** | Frustration, hostility, aggression |
| 🤢 | **Disgust** | Revulsion, strong disapproval |
| 😨 | **Fear** | Anxiety, worry, fright |
| 😊 | **Happy** | Joy, excitement, positivity |
| 😐 | **Neutral** | Calm, indifferent, composed |
| 😢 | **Sad** | Grief, depression, sorrow |
| 😲 | **Surprise** | Shock, astonishment, wonder |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive web application |
| **Backend** | Python 3.8+ | Core application logic |
| **Deep Learning** | TensorFlow / Keras | Emotion classification model |
| **Audio Processing** | Librosa | Feature extraction & analysis |
| **Visualization** | Matplotlib, Plotly | Charts, spectrograms, radar plots |
| **Live Recording** | Web MediaRecorder API | Browser microphone capture |
| **Deployment** | Streamlit Cloud | Cloud hosting |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/voice-mental-health-monitor.git
cd voice-mental-health-monitor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🚀 Usage

### Option A: Upload an Audio File

1. Click **"Browse files"** or drag and drop an audio file
2. Supported formats: **WAV, MP3, OGG, M4A, FLAC, WEBA**
3. View the waveform and Mel Spectrogram
4. Click **"Analyze Emotion"**
5. View the predicted emotion, confidence scores, and detailed charts
6. Read your personalized wellness suggestions

### Option B: Live Microphone Recording

1. Open the **"Microphone Recorder"** section
2. Click the microphone icon to **start recording**
3. Speak clearly for a few seconds
4. Click again to **stop recording**
5. Click **"Analyze Emotion"** to process

> 💡 **Tips for best results:**
> - Use a clear audio recording with minimal background noise
> - Recommended sample rate: **16 kHz or higher**
> - Speak naturally and expressively for accurate detection

---

## 📁 Project Structure

```
voice-mental-health-monitor/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 model.py                  # Deep learning model definition
├── 📄 feature_extraction.py     # Audio feature extraction pipeline
├── 📄 recommendations.py        # Emotion-based wellness suggestions
│
├── 📂 models/
│   └── emotion_model.h5         # Trained Keras model weights
│
├── 📂 utils/
│   ├── audio_utils.py           # Audio preprocessing utilities
│   └── visualization.py         # Chart and plot generation
│
├── 📂 assets/
│   └── style.css                # Custom styling (if any)
│
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
└── 📄 LICENSE                   # MIT License
```

---

## 📈 Results

### Test Results on TESS Dataset

| Audio Sample | True Emotion | Predicted Emotion | Confidence |
|---|---|---|---|
| OAF_base_disgust.wav | Disgust | ✅ **Disgust** | 100.0% |
| OAF_back_happy.wav | Happy | ✅ **Happy** | — |
| YAF_bought_happy.wav | Happy | ✅ **Happy** | — |
| Live recording (angry tone) | Angry | ✅ **Angry** | High |

### Analysis Views Available

- 📊 **Probability Distribution** — Horizontal bar chart of all 7 emotion probabilities
- 🕸️ **Confidence Radar** — Spider/radar chart showing multi-dimensional emotion space
- 🍩 **Top Emotions Donut** — Proportional ring chart of dominant emotions
- 📋 **Emotion Ranking Table** — Ranked list with statistical summary (min, max, avg, std dev)

---

## 🔮 Future Enhancements

- [ ] 🌐 **Multi-language support** — Emotion detection beyond English speech
- [ ] 📱 **Mobile App** — iOS/Android version using React Native
- [ ] 📈 **Emotion History** — Track and visualize emotion trends over time
- [ ] 🔗 **REST API** — Expose emotion analysis as an API endpoint
- [ ] 🤝 **Therapist Dashboard** — Clinician portal for patient monitoring
- [ ] 🧠 **Transformer Models** — Integrate wav2vec 2.0 / HuBERT for higher accuracy
- [ ] 🔒 **On-device Processing** — Privacy-first local inference
- [ ] 🔔 **Notification System** — Alerts for persistent negative emotional states

---

## 📚 Dataset

This project uses the **[TESS (Toronto Emotional Speech Set)](https://tspace.library.utoronto.ca/handle/1807/24487)** dataset:

- 📁 **2,800** audio stimuli in WAV format
- 👩 **2 actresses** — OAF (Older Adult Female) and YAF (Younger Adult Female)
- 🎭 **7 emotion** categories
- ⏱️ Duration: ~1–3 seconds per clip
- 🎙️ Sample rate: 24,414 Hz

> Additional datasets (RAVDESS, CREMA-D, SAVEE) can be combined for improved model robustness.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/AmazingFeature`
3. **Commit** your changes: `git commit -m 'Add AmazingFeature'`
4. **Push** to the branch: `git push origin feature/AmazingFeature`
5. **Open** a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct.

---

### 💜 Built with passion for mental health awareness

*If this project helped you, please consider giving it a ⭐ — it means a lot!*

<br/>

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/voice-mental-health-monitor?style=social)](https://github.com/YOUR_USERNAME/voice-mental-health-monitor)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/voice-mental-health-monitor?style=social)](https://github.com/YOUR_USERNAME/voice-mental-health-monitor)

<br/>

**🧠 Powered by Deep Learning &nbsp;•&nbsp; 🎤 Speech Emotion Recognition &nbsp;•&nbsp; 🚀 Built with Streamlit**

</div>
