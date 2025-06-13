## Create an app only in 14 lines of codes with Streamlit in python ##
import streamlit as st
from google import genai
st.title("Quick Start App")
Gemini_API_KEY = st.text_input("Enter your Gemini API Key", type="password")
if Gemini_API_KEY:
    st.session_state.client = genai.Client(Gemini_API_KEY)
    prompt = st.text_area("Enter your prompt")
    if st.button("Generate"):
        with st.spinner("Generating..."):
            response = st.session_state.client.chat(prompt=prompt, temperature=0.5, max_tokens=1000)
            st.write(response["message"]["content"])