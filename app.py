import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import io
import numpy as np
import streamlit as st
import librosa
import librosa.display
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Utilities.utils import extract_features
from tf_keras.models import load_model

# pip install audio-recorder-streamlit
from audio_recorder_streamlit import audio_recorder

# --- Page Configuration & CSS ---
st.set_page_config(page_title="Voice-Based Mental Health Monitoring System", page_icon="🎤", layout="wide")
st.markdown("""
<style>
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(12px); }
        60% { opacity: 1; transform: translateY(-6px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .suggestion-card { animation: fadeInUp 0.6s ease both; }
    .main-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .emotion-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }
    .result-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); }
    .metric-card { background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 10px; margin: 0.5rem 0; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
    .sidebar-content { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }
    .audio-player { background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.2); }
    .success-message { background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .emotion-emoji { font-size: 4rem; margin: 1rem 0; text-align: center; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
    .file-details { background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #667eea; }
    .recorder { text-align:center; padding:8px; }
    .rec-btn { padding:8px 14px; border-radius:8px; border:none; cursor:pointer; font-weight:600; margin:6px; }
</style>
""", unsafe_allow_html=True)

# --- Session state defaults ---
for key, val in {
    'analysis_complete': False,
    'audio_file': None,
    'results': None,
    'recorded_audio_bytes': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Single Model Loader (fixed & deduplicated) ---
@st.cache_resource
def load_all():
    try:
        model = load_model(
            os.path.join("model", "best_lstm_model.keras"),
            compile=False
        )
        scaler  = joblib.load(os.path.join("model", "scaler.joblib"))
        encoder = joblib.load(os.path.join("model", "encoder.joblib"))
        return model, scaler, encoder
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None

# --- Emotion suggestions & mapping ---
EMOTION_SUGGESTIONS = {
    "happy":   {"title": "You're sounding happy! 😊",  "tips": ["Keep this positive energy going!", "Write down something you're grateful for today.", "Share your happiness with someone close.", "Listen to your favourite upbeat song."]},
    "sad":     {"title": "You sound sad. 💙",           "tips": ["Take a deep breath and relax.", "Try a short calming exercise.", "Talk to a trusted friend or family member.", "Drink water and give yourself a small break."]},
    "angry":   {"title": "You sound angry. 😠",         "tips": ["Pause for a moment before reacting.", "Try slow breathing: inhale 4s → exhale 4s.", "Go for a short walk to cool down.", "Write down what bothered you to clear your mind."]},
    "fear":    {"title": "You seem anxious. 😰",        "tips": ["Focus on your surroundings and be present.", "Use the 5-4-3-2-1 grounding technique.", "Talk to someone you trust about your worry.", "Try stretching or light movement."]},
    "surprise":{"title": "You sound surprised! 😮",    "tips": ["Take a moment to understand the cause.", "Reflect on whether the surprise was positive or negative.", "Talk it out with someone if needed.", "Try re-listening to your recording for clarity."]},
    "neutral": {"title": "Neutral tone detected. 😐",  "tips": ["Add more expression if you're trying to convey emotion.", "Relax your throat and try speaking again.", "Record again with intentional tone shifts.", "Smile lightly while speaking to change tone."]},
    "disgust": {"title": "Disgust detected. 🤢",        "tips": ["Take a moment to breathe.", "Try reframing the situation more calmly.", "Step away briefly to reset your mindset.", "Drink some water and relax your facial muscles."]}
}
emotion_mapping = {
    'angry':   {'emoji': '😠', 'color': '#FF4444'},
    'disgust': {'emoji': '🤢', 'color': '#4CAF50'},
    'fear':    {'emoji': '😨', 'color': '#FF9800'},
    'happy':   {'emoji': '😊', 'color': '#4CAF50'},
    'neutral': {'emoji': '😐', 'color': '#9E9E9E'},
    'sad':     {'emoji': '😢', 'color': '#2196F3'},
    'surprise':{'emoji': '😲', 'color': '#9C27B0'}
}

# --- Visualization helpers ---
def create_bar_chart(emotions, probabilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_data = sorted(zip(emotions, probabilities), key=lambda x: x[1])
    sorted_emotions, sorted_probs = zip(*sorted_data)
    colors = [emotion_mapping.get(e, {'color': '#666666'})['color'] for e in sorted_emotions]
    bars = ax.barh(range(len(sorted_emotions)), sorted_probs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_emotions)))
    ax.set_yticklabels([f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" for e in sorted_emotions])
    ax.set_xlabel('Confidence Score')
    ax.set_title('Emotion Probability Distribution', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.1%}', va='center', fontweight='bold')
    plt.tight_layout()
    return fig

def create_radar_chart(emotions, probabilities):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    probs = list(probabilities) + [probabilities[0]]
    angles += angles[:1]
    ax.plot(angles, probs, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, probs, color='#667eea', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" for e in emotions])
    ax.set_ylim(0, 1)
    ax.set_title('Emotion Confidence Radar', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True)
    return fig

def create_donut_chart(emotions, probabilities):
    fig, ax = plt.subplots(figsize=(8, 8))
    top_indices = np.argsort(probabilities)[-5:][::-1]
    top_emotions = [emotions[i] for i in top_indices]
    top_probs    = [probabilities[i] for i in top_indices]
    colors = [emotion_mapping.get(e, {'color': '#666666'})['color'] for e in top_emotions]
    ax.pie(top_probs,
           labels=[f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" for e in top_emotions],
           colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.add_artist(Circle((0, 0), 0.70, fc='white'))
    ax.set_title('Top 5 Emotions Distribution', fontsize=16, fontweight='bold')
    return fig

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr):
    S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    ax.set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

# --- Analyze audio ---
def analyze_audio_bytesio(bio: io.BytesIO, model, scaler, encoder):
    try:
        bio.seek(0)
        y, sr = librosa.load(bio, sr=None)
        if y.ndim > 1:
            y = librosa.to_mono(y)
    except Exception as e:
        st.error(f"Failed to load audio: {e}")
        return

    with st.spinner("🔍 Analyzing audio features..."):
        progress_bar = st.progress(0)
        try:
            progress_bar.progress(10)
            features = extract_features(y, sr).reshape(1, -1)
            progress_bar.progress(50)
            features = scaler.transform(features)
            progress_bar.progress(75)
            features = np.expand_dims(features, axis=2)
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction)

            if hasattr(encoder, 'categories_'):
                predicted_label = encoder.categories_[0][predicted_class]
                all_emotions    = encoder.categories_[0]
            elif hasattr(encoder, 'classes_'):
                predicted_label = encoder.classes_[predicted_class]
                all_emotions    = encoder.classes_
            else:
                predicted_label = str(predicted_class)
                all_emotions    = [str(i) for i in range(len(prediction[0]))]

            progress_bar.progress(100)
            st.session_state.results = {
                'predicted_emotion': predicted_label,
                'probabilities': prediction[0],
                'emotions': all_emotions
            }
            st.session_state.analysis_complete = True
            st.markdown('<div class="success-message">✅ Analysis Complete! Results are ready.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            progress_bar.empty()

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🎤 Voice-Based Mental Health Monitoring System</h1>
    <p>AI-Powered Emotion Analysis from Speech Patterns</p>
    <p>Upload or record an audio file to discover the emotional tone</p>
</div>
""", unsafe_allow_html=True)

# --- Load models (single call) ---
model, scaler, encoder = load_all()
if model is None:
    st.error("⚠️ Models could not be loaded. Please check your model files in the /model folder.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h3>🎯 How it Works</h3>
        <p>Our AI analyzes audio features like:</p>
        <ul>
            <li>🎵 Spectral characteristics</li>
            <li>🔊 Voice energy patterns</li>
            <li>📊 Frequency distributions</li>
            <li>⚡ Temporal dynamics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-content">
        <h3>📋 Supported Formats</h3>
        <p>• WAV / MP3 / OGG / M4A / FLAC</p>
        <p>• Duration: 1–120 seconds</p>
        <p>• Sample rate: 16 kHz+ recommended</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Reset Analysis", help="Clear all results and start over"):
        st.session_state.analysis_complete   = False
        st.session_state.audio_file          = None
        st.session_state.results             = None
        st.session_state.recorded_audio_bytes = None
        st.rerun()

# --- Main layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="emotion-card">
        <h2>📁 Upload or Record Audio</h2>
        <p>Choose a clear audio recording with speech to analyze emotional content.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "weba", "mp3", "ogg", "m4a", "flac"],
        help="Select an audio file containing speech for emotion analysis"
    )

    st.markdown("""
    <div class="audio-player">
        <h4>🎙️ Microphone Recorder</h4>
        <p>Click the microphone button to start recording, click again to stop.
        Allow microphone access when prompted by your browser.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Open Microphone Recorder", expanded=False):
        recorded_bytes = audio_recorder(
            text="Click to record",
            recording_color="#FF4444",
            neutral_color="#667eea",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=3.0,
            sample_rate=16_000,
        )
        if recorded_bytes:
            if len(recorded_bytes) > 1024:
                st.session_state.recorded_audio_bytes = recorded_bytes
                st.success("✅ Recording captured! Scroll down and click **Analyze Emotion**.")
            else:
                st.info("Recording seems too short — please speak for at least 1 second.")

    # Determine active audio source
    active_audio = None
    audio_name   = None
    buf          = None

    if uploaded_file is not None:
        active_audio = uploaded_file
        audio_name   = getattr(uploaded_file, "name", "uploaded_audio")
    elif st.session_state.recorded_audio_bytes is not None:
        active_audio      = io.BytesIO(st.session_state.recorded_audio_bytes)
        active_audio.name = "recording.wav"
        audio_name        = "recording.wav"

    if active_audio is not None:
        st.markdown(
            f"<div class='file-details'><h4>📄 File Information</h4>"
            f"<p><strong>Name:</strong> {audio_name}</p></div>",
            unsafe_allow_html=True
        )
        try:
            active_audio.seek(0)
            audio_bytes = active_audio.read()
            st.audio(audio_bytes)

            buf      = io.BytesIO(audio_bytes)
            buf.name = audio_name
            y, sr    = librosa.load(io.BytesIO(audio_bytes), sr=None)
            if y.ndim > 1:
                y = librosa.to_mono(y)

            st.markdown("### 🔎 Audio Visualizations")
            with st.expander("Show Waveform and Spectrogram", expanded=True):
                wfig = plot_waveform(y, sr)
                st.pyplot(wfig)
                plt.close(wfig)
                sfig = plot_spectrogram(y, sr)
                st.pyplot(sfig)
                plt.close(sfig)

        except Exception as e:
            st.warning(f"Could not load audio for playback/visualization: {e}")
            buf = None

        if st.button("🧠 Analyze Emotion", key="analyze"):
            if buf is None:
                st.error("No audio buffer available for analysis.")
            else:
                analyze_audio_bytesio(buf, model, scaler, encoder)

with col2:
    if st.session_state.analysis_complete and st.session_state.results:
        results           = st.session_state.results
        predicted_emotion = results['predicted_emotion']
        probabilities     = results['probabilities']
        emotions          = results['emotions']
        emotion_info      = emotion_mapping.get(predicted_emotion, {'emoji': '❓', 'color': '#666666'})

        st.markdown(f"""
        <div class="result-card">
            <div class="emotion-emoji">{emotion_info['emoji']}</div>
            <h2>Predicted Emotion</h2>
            <h1 style="font-size: 2.5rem; margin: 1rem 0;">{predicted_emotion.upper()}</h1>
            <p style="font-size: 1.2rem;">Confidence: {probabilities[np.argmax(probabilities)]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="emotion-card"><h3>🎯 Confidence Scores</h3></div>', unsafe_allow_html=True)
        for emotion, prob in zip(emotions, probabilities):
            ei = emotion_mapping.get(emotion, {'emoji': '❓', 'color': '#666666'})
            c1, c2, c3, c4 = st.columns([1, 2, 4, 1])
            with c1: st.markdown(f"<div style='font-size:2rem;text-align:center'>{ei['emoji']}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"**{emotion.capitalize()}**")
            with c3: st.progress(float(prob))
            with c4: st.markdown(f"**{prob:.1%}**")

# --- Emotion Suggestions ---
if st.session_state.analysis_complete and st.session_state.results:
    emotion_key = st.session_state.results['predicted_emotion']
    suggestion  = EMOTION_SUGGESTIONS.get(emotion_key)
    if suggestion:
        color = emotion_mapping.get(emotion_key, {'color': '#667eea'})['color']
        emoji = emotion_mapping.get(emotion_key, {'emoji': '💡'})['emoji']
        parts = [
            f"<div class='suggestion-card' style='border-radius:12px;padding:20px;margin:12px 0;background:linear-gradient(135deg,{color}22 0%,#ffffff 100%);box-shadow:0 4px 12px rgba(0,0,0,0.08);'>",
            "<div style='display:flex;align-items:center;gap:12px;'>",
            f"<div style='font-size:3rem;'>{emoji}</div><div>",
            f"<h3 style='margin:0 0 6px 0;'>{suggestion['title']}</h3>",
            "<p style='margin:0;color:#333;'>Here are some quick suggestions you can try:</p>",
            "</div></div>",
            "<div style='margin-top:15px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;'>",
        ]
        for tip in suggestion['tips']:
            safe = tip.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            parts.append(f"<div style='background:#fff;border-radius:10px;padding:10px 14px;margin:10px 0;box-shadow:0 3px 8px rgba(0,0,0,0.06);'><div style='font-weight:600;color:#111;'>🔸 {safe}</div></div>")
        parts += ["</div>", "</div>"]
        st.markdown(''.join(parts), unsafe_allow_html=True)

# --- Detailed Analysis ---
if st.session_state.analysis_complete and st.session_state.results:
    st.markdown("---")
    st.markdown("## 📊 Detailed Analysis")
    emotions      = st.session_state.results['emotions']
    probabilities = st.session_state.results['probabilities']

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Distribution", "🎯 Confidence Radar", "🍩 Top Emotions", "📈 Emotion Ranking"])
    with tab1:
        st.markdown("### Horizontal Bar Chart")
        fig = create_bar_chart(emotions, probabilities); st.pyplot(fig); plt.close(fig)
    with tab2:
        st.markdown("### Radar Chart")
        fig = create_radar_chart(emotions, probabilities); st.pyplot(fig); plt.close(fig)
    with tab3:
        st.markdown("### Donut Chart - Top 5 Emotions")
        fig = create_donut_chart(emotions, probabilities); st.pyplot(fig); plt.close(fig)
    with tab4:
        st.markdown("### Emotion Ranking Table")
        df = pd.DataFrame({
            'Rank':       range(1, len(emotions)+1),
            'Emotion':    emotions,
            'Confidence': probabilities,
            'Emoji':      [emotion_mapping.get(e, {'emoji':'❓'})['emoji'] for e in emotions]
        }).sort_values('Confidence', ascending=False).reset_index(drop=True)
        df['Rank']       = range(1, len(df)+1)
        df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📈 Statistical Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Highest Confidence", f"{np.max(probabilities):.1%}")
        c2.metric("Lowest Confidence",  f"{np.min(probabilities):.1%}")
        c3.metric("Average Confidence", f"{np.mean(probabilities):.1%}")
        c4.metric("Std. Deviation",     f"{np.std(probabilities):.1%}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:2rem;">
    <p>🧠 Powered by Deep Learning • 🎤 Voice-Based Mental Health Monitoring System • 🚀 Built with Streamlit</p>
    <p>Upload clear audio recordings for best results</p>
</div>
""", unsafe_allow_html=True)
