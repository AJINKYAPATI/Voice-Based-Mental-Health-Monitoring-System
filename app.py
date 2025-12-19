import io
import base64
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import librosa
import librosa.display
from joblib import load
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Utilities.utils import extract_features

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
    'audio_file': None,      # UploadedFile or BytesIO
    'results': None,
    'recorded_b64': None,    # base64 from mic component
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Model loader ---
@st.cache_resource
def load_models():
    try:
        MODEL_PATH = "model/best_lstm_model.keras"
        SCALER_PATH = "model/scaler.joblib"
        ENCODER_PATH = "model/encoder.joblib"
        model = load_model(MODEL_PATH)
        scaler = load(SCALER_PATH)
        encoder = load(ENCODER_PATH)
        return model, scaler, encoder
    except Exception as e:
        return None, None, str(e)

# --- Emotion suggestions & mapping (unchanged) ---
EMOTION_SUGGESTIONS = {
    "happy": {"title": "You're sounding happy! 😊", "tips": ["Keep this positive energy going!", "Write down something you're grateful for today.", "Share your happiness with someone close.", "Listen to your favourite upbeat song."]},
    "sad": {"title": "You sound sad. 💙", "tips": ["Take a deep breath and relax.", "Try a short calming exercise.", "Talk to a trusted friend or family member.", "Drink water and give yourself a small break."]},
    "angry": {"title": "You sound angry. 😠", "tips": ["Pause for a moment before reacting.", "Try slow breathing: inhale 4s → exhale 4s.", "Go for a short walk to cool down.", "Write down what bothered you to clear your mind."]},
    "fear": {"title": "You seem anxious. 😰", "tips": ["Focus on your surroundings and be present.", "Use the 5-4-3-2-1 grounding technique.", "Talk to someone you trust about your worry.", "Try stretching or light movement."]},
    "surprise": {"title": "You sound surprised! 😮", "tips": ["Take a moment to understand the cause.", "Reflect on whether the surprise was positive or negative.", "Talk it out with someone if needed.", "Try re-listening to your recording for clarity."]},
    "neutral": {"title": "Neutral tone detected. 😐", "tips": ["Add more expression if you're trying to convey emotion.", "Relax your throat and try speaking again.", "Record again with intentional tone shifts.", "Smile lightly while speaking to change tone."]},
    "disgust": {"title": "Disgust detected. 🤢", "tips": ["Take a moment to breathe.", "Try reframing the situation more calmly.", "Step away briefly to reset your mindset.", "Drink some water and relax your facial muscles."]}
}
emotion_mapping = {
    'angry': {'emoji': '😠', 'color': '#FF4444'},
    'disgust': {'emoji': '🤢', 'color': '#4CAF50'},
    'fear': {'emoji': '😨', 'color': '#FF9800'},
    'happy': {'emoji': '😊', 'color': '#4CAF50'},
    'neutral': {'emoji': '😐', 'color': '#9E9E9E'},
    'sad': {'emoji': '😢', 'color': '#2196F3'},
    'surprise': {'emoji': '😲', 'color': '#9C27B0'}
}

# --- Visualization helpers (kept intact) ---
def create_bar_chart(emotions, probabilities):
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_data = sorted(zip(emotions, probabilities), key=lambda x: x[1])
    sorted_emotions, sorted_probs = zip(*sorted_data)
    colors = [emotion_mapping.get(emotion, {'color': '#666666'})['color'] for emotion in sorted_emotions]
    bars = ax.barh(range(len(sorted_emotions)), sorted_probs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_emotions)))
    ax.set_yticklabels([f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" for e in sorted_emotions])
    ax.set_xlabel('Confidence Score')
    ax.set_title('Emotion Probability Distribution', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.1%}', va='center', fontweight='bold')
    plt.tight_layout()
    return fig

def create_radar_chart(emotions, probabilities):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    probabilities = list(probabilities) + [probabilities[0]]
    angles += angles[:1]
    ax.plot(angles, probabilities, 'o-', linewidth=2, color='#667eea')
    ax.fill(angles, probabilities, color='#667eea', alpha=0.25)
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
    top_probs = [probabilities[i] for i in top_indices]
    colors = [emotion_mapping.get(emotion, {'color': '#666666'})['color'] for emotion in top_emotions]
    wedges, texts, autotexts = ax.pie(top_probs, labels=[f"{emotion_mapping.get(e, {'emoji': '❓'})['emoji']} {e.capitalize()}" for e in top_emotions], colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    centre_circle = Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title('Top 5 Emotions Distribution', fontsize=16, fontweight='bold')
    return fig

def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr, title="Mel Spectrogram (dB)"):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

# --- Helper: analyze bytesIO audio ---
def analyze_audio_bytesio(bio: io.BytesIO, model, scaler, encoder):
    """Load bytesIO into numpy, extract features, run model pipeline, store results in session_state."""
    try:
        bio.seek(0)
        y, sr = librosa.load(bio, sr=None)
        if y.ndim > 1:
            y = librosa.to_mono(y)
    except Exception as e:
        st.error(f"Failed to load audio for analysis: {e}")
        return

    # Run pipeline with progress
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
            elif hasattr(encoder, 'classes_'):
                predicted_label = encoder.classes_[predicted_class]
            else:
                predicted_label = str(predicted_class)
            progress_bar.progress(100)

            st.session_state.results = {
                'predicted_emotion': predicted_label,
                'probabilities': prediction[0],
                'emotions': encoder.categories_[0] if hasattr(encoder, 'categories_') else encoder.classes_
            }
            st.session_state.analysis_complete = True
            st.markdown("""
            <div class="success-message">
                  Analysis Complete! Results are ready.
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            progress_bar.empty()

# --- Microphone recorder component ---
def mic_recorder_component():
    """
    Returns a base64 string of recorded audio (webm) or None.
    Uses MediaRecorder in browser and Streamlit.setComponentValue to send b64 back to Python.
    """
    RECORD_HTML = """
    <div class="recorder">
      <div id="status">Click <strong>Start</strong> to record (browser will ask for mic permission).</div>
      <div style="margin-top:8px;">
        <button id="startBtn" class="rec-btn" style="background:#4CAF50;color:white">Start</button>
        <button id="stopBtn" class="rec-btn" style="background:#FF4444;color:white" disabled>Stop</button>
        <button id="downloadBtn" class="rec-btn" style="background:#2196F3;color:white" disabled>Download</button>
      </div>
      <div style="margin-top:12px;">
        <audio id="player" controls></audio>
      </div>
      <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const player = document.getElementById('player');
        const status = document.getElementById('status');
        let mediaRecorder;
        let audioChunks = [];

        function toBase64(blob) {
          return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
          });
        }

        startBtn.onclick = async () => {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
              const blob = new Blob(audioChunks, { type: 'audio/webm' });
              // play
              const url = URL.createObjectURL(blob);
              player.src = url;
              player.play();
              // enable download
              downloadBtn.disabled = false;
              // convert to base64 and send to Streamlit
              const b64 = await toBase64(blob);
              // send the base64 string back to Streamlit (component value)
              const Send = (value) => {
                // Streamlit.setComponentValue is available in components embedded iframe
                if (typeof Streamlit !== "undefined" && Streamlit.setComponentValue) {
                  Streamlit.setComponentValue(value)
                } else {
                  // fallback: postMessage - this won't set the component return but prevents breaking
                  window.parent.postMessage({ type: 'mic-record', data: value }, '*');
                }
              }
              Send(b64);
            };
            mediaRecorder.start();
            status.innerHTML = "Recording... Press Stop when finished.";
            startBtn.disabled = true;
            stopBtn.disabled = false;
          } catch (err) {
            status.innerHTML = "Microphone access denied or not available.";
            console.error(err);
          }
        };

        stopBtn.onclick = () => {
          if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            stopBtn.disabled = true;
            startBtn.disabled = false;
            status.innerHTML = "Processing recording...";
          }
        };

        downloadBtn.onclick = () => {
          // create a blob from current audioChunks and download
          const blob = new Blob(audioChunks, { type: 'audio/webm' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = 'recording.webm';
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
        };
      </script>
    </div>
    """
    try:
        result = components.html(RECORD_HTML, height=220)
        # result will be the value passed to Streamlit.setComponentValue in the JS (base64 string)
        return result
    except Exception as e:
        st.warning(f"Microphone component not available: {e}")
        return None

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🎤 Voice-Based Mental Health Monitoring System</h1>
    <p>AI-Powered Emotion Analysis from Speech Patterns</p>
    <p>Upload or record an audio file to discover the emotional tone</p>
</div>
""", unsafe_allow_html=True)

# --- Load models ---
model, scaler, encoder = load_models()
if model is None:
    st.error("⚠️ Models could not be loaded. Please check your model paths.")
    if isinstance(encoder, str):
        st.error(f"Loader error: {encoder}")
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
        <p>• WAV/MP3/OGG/M4A/FLAC (recommended)</p>
        <p>• Duration: 1-120 seconds</p>
        <p>• Sample rate: 16kHz+ recommended</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Reset Analysis", help="Clear all results and start over"):
        st.session_state.analysis_complete = False
        st.session_state.audio_file = None
        st.session_state.results = None
        st.session_state.recorded_b64 = None
        st.experimental_rerun()


# --- Main layout ---
col1, col2 = st.columns([2, 1])


with col1:
    st.markdown("""
    <div class="emotion-card">
        <h2>📁 Upload or Record Audio</h2>
        <p>Choose a clear audio recording with speech to analyze emotional content.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav","weba", "mp3", "ogg", "m4a", "flac"], help="Select an audio file containing speech for emotion analysis")

    st.markdown("""
    <div class="audio-player">
        <h4>🎙️ Microphone Recorder</h4>
        <p>Use your browser microphone to record a short sample (recommended 1-30s). Click <strong>Start</strong>, speak, then <strong>Stop</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Microphone recorder component — returns base64 audio (webm)
    with st.expander("Open Microphone Recorder", expanded=False):
        mic_b64 = mic_recorder_component()
        # If the recorder returned a base64 string, store it in session_state for later
        if mic_b64:
            st.session_state.recorded_b64 = mic_b64
            st.success("✅ Recording received (from microphone). You can play it below or Analyze Emotion.")

    # Choose active audio: uploaded has precedence over recorded
    active_audio = None
    audio_name = None
    if uploaded_file is not None:
        st.session_state.audio_file = uploaded_file
        active_audio = uploaded_file
        audio_name = getattr(uploaded_file, "name", "uploaded_audio")
    elif st.session_state.recorded_b64 is not None:
        # convert base64 to bytes (we expect webm)
        try:
            b64 = st.session_state.recorded_b64
            data = base64.b64decode(b64)
            buf = io.BytesIO(data)
            buf.name = "recording.webm"
            st.session_state.audio_file = buf
            active_audio = buf
            audio_name = buf.name
        except Exception as e:
            st.warning(f"Could not decode recorded audio: {e}")
            active_audio = None

    if active_audio is not None:
        st.markdown(f"<div class='file-details'><h4>📄 File Information</h4><p><strong>Name:</strong> {audio_name}</p></div>", unsafe_allow_html=True)
        try:
            # For uploaded files (streamlit UploadedFile) or BytesIO
            if hasattr(active_audio, "read"):
                active_audio.seek(0)
                audio_bytes = active_audio.read()
                st.audio(audio_bytes)
                # Prepare buffer for visualization & analysis
                buf = io.BytesIO(audio_bytes)
                buf.name = audio_name
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            else:
                st.warning("Active audio object is not readable.")
                buf = None
                y = None
                sr = None

            if y is not None:
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

        # Analyze button
        if st.button("🧠 Analyze Emotion", key="analyze"):
            if buf is None:
                st.error("No audio buffer available for analysis.")
            else:
                analyze_audio_bytesio(buf, model, scaler, encoder)

with col2:
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        predicted_emotion = results['predicted_emotion']
        probabilities = results['probabilities']
        emotions = results['emotions']
        emotion_info = emotion_mapping.get(predicted_emotion, {'emoji': '❓', 'color': '#666666'})
        result_html = f"""
        <div class="result-card">
            <div class="emotion-emoji">{emotion_info['emoji']}</div>
            <h2>Predicted Emotion</h2>
            <h1 style="font-size: 2.5rem; margin: 1rem 0;">{predicted_emotion.upper()}</h1>
            <p style="font-size: 1.2rem;">Confidence: {probabilities[np.argmax(probabilities)]:.1%}</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        st.markdown("""
        <div class="emotion-card">
            <h3>🎯 Confidence Scores</h3>
        </div>
        """, unsafe_allow_html=True)
        for emotion, prob in zip(emotions, probabilities):
            emotion_info = emotion_mapping.get(emotion, {'emoji': '❓', 'color': '#666666'})
            col_emoji, col_name, col_bar, col_percent = st.columns([1, 2, 4, 1])
            with col_emoji:
                st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{emotion_info['emoji']}</div>", unsafe_allow_html=True)
            with col_name:
                st.markdown(f"**{emotion.capitalize()}**")
            with col_bar:
                st.progress(float(prob))
            with col_percent:
                st.markdown(f"**{prob:.1%}**")


# --- Emotion Suggestions Display ---

if st.session_state.analysis_complete and st.session_state.results:
    emotion_key = st.session_state.results['predicted_emotion']
    suggestion = EMOTION_SUGGESTIONS.get(emotion_key, None)

    if suggestion:
        color = emotion_mapping.get(emotion_key, {'color': '#667eea'})['color']
        emoji = emotion_mapping.get(emotion_key, {'emoji': '💡'})['emoji']

        parts = []
        parts.append(f"<div class='suggestion-card' style='border-radius:12px; padding:20px; margin:12px 0; background: linear-gradient(135deg, {color}22 0%, #ffffff 100%); box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>")
        parts.append("<div style='display:flex; align-items:center; gap:12px;'>")
        parts.append(f"<div style='font-size:3rem;'>{emoji}</div>")
        parts.append("<div>")
        parts.append(f"<h3 style='margin:0 0 6px 0;'>{suggestion['title']}</h3>")
        parts.append("<p style='margin:0; color:#333;'>Here are some quick suggestions you can try:</p>")
        parts.append("</div></div>")
        parts.append("<div style='margin-top:15px; display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:10px;'>")
        for tip in suggestion['tips']:
            safe_tip = str(tip).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            parts.append(
                "<div style='background:#ffffff; border-radius:10px; padding:10px 14px; margin:10px 0; box-shadow:0 3px 8px rgba(0,0,0,0.06);'>"
                f"<div style='font-weight:600; color:#111;'>🔸 {safe_tip}</div>"
                "</div>"
            )
        parts.append("</div>")
        parts.append("</div>")

        card_html = ''.join(parts)
        st.markdown(card_html, unsafe_allow_html=True)

# Results visualization (full width)
if st.session_state.analysis_complete and st.session_state.results:
    st.markdown("---")
    st.markdown("## 📊 Detailed Analysis")
    results = st.session_state.results
    emotions = results['emotions']
    probabilities = results['probabilities']
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Distribution", "🎯 Confidence Radar", "🍩 Top Emotions", "📈 Emotion Ranking"]) 
    with tab1:
        st.markdown("### Horizontal Bar Chart")
        fig = create_bar_chart(emotions, probabilities)
        st.pyplot(fig)
        plt.close(fig)
    with tab2:
        st.markdown("### Radar Chart")
        fig = create_radar_chart(emotions, probabilities)
        st.pyplot(fig)
        plt.close(fig)
    with tab3:
        st.markdown("### Donut Chart - Top 5 Emotions")
        fig = create_donut_chart(emotions, probabilities)
        st.pyplot(fig)
        plt.close(fig)
    with tab4:
        st.markdown("### Emotion Ranking Table")
        df_ranking = pd.DataFrame({'Rank': range(1, len(emotions) + 1),'Emotion': emotions,'Confidence': probabilities,'Emoji': [emotion_mapping.get(e, {'emoji': '❓'})['emoji'] for e in emotions]}).sort_values('Confidence', ascending=False).reset_index(drop=True)
        df_ranking['Rank'] = range(1, len(df_ranking) + 1)
        df_ranking['Confidence'] = df_ranking['Confidence'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_ranking,use_container_width=True)
        st.markdown("### 📈 Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Highest Confidence", f"{np.max(probabilities):.1%}")
        with col2:
            st.metric("Lowest Confidence", f"{np.min(probabilities):.1%}")
        with col3:
            st.metric("Average Confidence", f"{np.mean(probabilities):.1%}")
        with col4:
            st.metric("Std. Deviation", f"{np.std(probabilities):.1%}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 Powered by Deep Learning • 🎤 Voice-Based Mental Health Monitoring System • 🚀 Built with Streamlit</p>
    <p>Upload clear audio recordings for best results</p>
</div>
""", unsafe_allow_html=True)
