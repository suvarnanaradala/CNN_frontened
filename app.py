import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="InstruNet AI",
    layout="wide"
)

# Title
st.title("🎵 InstruNet AI")
st.subheader("CNN-Based Music Instrument Recognition System")

st.write("Upload an audio file to detect musical instruments")

# Audio upload
audio = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"]
)

if audio is not None:

    # Play audio
    st.audio(audio)
    st.success("Audio uploaded successfully!")

    # Load audio
    y, sr = librosa.load(audio, sr=None)

    # ---------------- Waveform ----------------
    st.subheader("Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # ---------------- Spectrogram ----------------
    st.subheader("Mel Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(
        S_dB,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax2
    )
    fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
    st.pyplot(fig2)

    # ---------------- Instrument Output (Dummy) ----------------
    st.subheader("Detected Instruments")

    instruments = {
        "Piano": 0.85,
        "Drums": 0.72,
        "Guitar": 0.68,
        "Bass": 0.80,
        "Violin": 0.25
    }

    for instrument, confidence in instruments.items():
        st.write(f"{instrument} : {int(confidence * 100)}%")
        st.progress(confidence)

    # ---------------- Intensity ----------------
    st.subheader("Instrument Intensity Over Time")
    st.text("Piano   : |||||||||||||||||")
    st.text("Drums   : |||||||||||")
    st.text("Guitar  : |||||||||")
    st.text("Bass    : ||||||||||||")

    st.success("Instrument analysis completed successfully!")
