import streamlit as st
import pyttsx3
import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os


# Set path to your local ffmpeg.exe
ffmpeg_path = r"C:\Users\saima\OneDrive\Desktop\Data Science\ffmpeg-2025-05-26-git-43a69886b2-essentials_build"  # <-- change this to your actual path
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)


# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Set up pyttsx3 engine once
@st.cache_resource
def get_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

tts_engine = get_tts_engine()

# Simple rule-based chatbot
def get_bot_response(user_input):
    if "hello" in user_input.lower():
        return "Hello! How can I assist you?"
    elif "how are you" in user_input.lower():
        return "I'm just a chatbot, but thanks for asking!"
    else:
        return "I'm not sure how to respond to that yet."

# Speak with pyttsx3
def speak_text(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tts_engine.save_to_file(text, f.name)
        tts_engine.runAndWait()
        return f.name

# Record voice input
def record_audio(duration=5, samplerate=16000):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, audio, samplerate)
        return f.name

# Transcribe voice with Whisper
def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]

# Streamlit UI
st.title("🤖 Fast Chatbot with Voice & Text")

mode = st.radio("Choose input mode:", ["Text", "Voice"])
user_input = ""

if mode == "Text":
    user_input = st.text_input("Type your message:")
    if st.button("Send") and user_input:
        response = get_bot_response(user_input)
        st.markdown(f"**Bot:** {response}")
        audio_path = speak_text(response)
        st.audio(audio_path, format='audio/wav')

elif mode == "Voice":
    if st.button("Record and Send"):
        audio_file = record_audio()
        user_input = transcribe_audio(audio_file)
        st.markdown(f"**You:** {user_input}")
        response = get_bot_response(user_input)
        st.markdown(f"**Bot:** {response}")
        audio_path = speak_text(response)
        st.audio(audio_path, format='audio/wav')

def speak_text(text):
    file_path = "output_audio.wav"
    tts_engine.save_to_file(text, file_path)
    tts_engine.runAndWait()
    return file_path


