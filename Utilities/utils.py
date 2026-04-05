import numpy as np
import librosa

def extract_features(data, sample_rate):
    result = []
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=data)
    result.append(np.mean(zcr))
    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    result.extend(np.mean(chroma, axis=1))
    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    result.extend(np.mean(mfcc, axis=1))
    # RMS
    rms = librosa.feature.rms(y=data)
    result.append(np.mean(rms))
    return np.array(result)
