import streamlit as st
import os
from src.nlp_engine import MedicalNLP

st.set_page_config(page_title="MedScribe AI", page_icon="🩺", layout="wide")

@st.cache_resource
def load_engine():
    return MedicalNLP()

def main():
    st.title("🩺 AI Physician Notetaker")
    
    with st.spinner("🚀 Loading AI Models..."):
        engine = load_engine()

    tab1, tab2 = st.tabs(["📄 Text Input", "🎤 Audio Upload"])

    # TEXT TAB
    with tab1:
        default_text = "Physician: Good morning, Ms. Janet Jones.\nPatient: I have a headache and I took Paracetamol."
        text_in = st.text_area("Transcript", value=default_text, height=150)
        
        if st.button("Analyze Text", type="primary"):
            process(engine, text_in)

    # AUDIO TAB
    with tab2:
        audio = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
        if audio and st.button("Transcribe & Analyze"):
            with open("temp.mp3", "wb") as f:
                f.write(audio.getbuffer())
            
            with st.spinner("Transcribing..."):
                text = engine.transcribe("temp.mp3")
                st.info(f"Transcript: {text}")
                process(engine, text)
            os.remove("temp.mp3")

def process(engine, text):
    results = engine.analyze(text)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("📋 Medical Summary")
        st.json(results["Medical_Summary"])
    with col2:
        st.subheader("💭 Sentiment Analysis")
        st.json(results["Sentiment_Analysis"])
    with col3:
        st.subheader("📝 SOAP Note")
        st.json(results["SOAP_Note"])

if __name__ == "__main__":
    main()