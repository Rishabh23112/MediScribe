# 🩺 MediScribe - AI Physician Notetaker

An intelligent medical transcription and NLP pipeline that converts physician-patient conversations into structured clinical documentation.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Audio Transcription** | Convert MP3/WAV audio to text using OpenAI Whisper |
| 🔬 **Medical Entity Extraction** | Extract symptoms, diagnoses, treatments using BioBERT |
| 💭 **Sentiment Analysis** | Detect patient emotional state and intent |
| 📝 **SOAP Note Generation** | Auto-generate structured clinical notes |
| 🌐 **Web Interface** | Interactive Streamlit dashboard |

---

## 🧠 Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| `d4data/biomedical-ner-all` | Medical Named Entity Recognition (NER) | HuggingFace |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment Analysis | HuggingFace |
| `en_core_web_sm` | General NER (Person names, dates) | spaCy |
| `bert-extractive-summarizer` | Extractive text summarization | BERT |
| `openai-whisper` (base) | Speech-to-Text transcription | OpenAI |

---

## 📁 Project Structure

```
MediScribe/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
└── src/
    ├── __init__.py
    ├── config.py          # Model configurations & device setup
    ├── nlp_engine.py      # Core NLP pipeline
    └── main.py            # CLI interface
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Rishabh23112/MediScribe.git
cd MediScribe
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the Application

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Command Line:**
```bash
python -m src.main
```

---

## 📋 Output Format

### Medical Summary
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Headache", "Nausea"],
  "Diagnosis": ["Migraine"],
  "Treatment": ["Sumatriptan", "Paracetamol"],
  "Prognosis": "I hope to recover in a few days"
}
```

### Sentiment Analysis
```json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}
```

### SOAP Note
```json
{
  "Subjective": {
    "Chief_Complaint": "Headache, Nausea",
    "History_of_Present_Illness": "Patient reports severe headache..."
  },
  "Objective": {
    "Physical_Exam": "Vital signs stable.",
    "Observations": "Patient is alert and oriented."
  },
  "Assessment": {
    "Diagnosis": "Migraine",
    "Severity": "Moderate"
  },
  "Plan": {
    "Treatment": "Sumatriptan, Paracetamol",
    "Follow-Up": "Recovery expected in a few days"
  }
}
```

---

## ⚙️ Configuration

Device configuration is handled automatically in `src/config.py`:

| Device | Condition |
|--------|-----------|
| `cuda` | NVIDIA GPU with CUDA available |
| `mps` | Apple Silicon (M1/M2) |
| `cpu` | Fallback for all other systems |

---

## 📦 Requirements

```
torch
transformers
bert-extractive-summarizer
spacy
openai-whisper
python-dotenv
streamlit
accelerate
sentencepiece
numpy
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `spacy model not found` | Run `python -m spacy download en_core_web_sm` |
| CUDA out of memory | Set `WHISPER_SIZE = "tiny"` in `config.py` |
| Slow on CPU | Models run faster on GPU; use smaller Whisper model |

---

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

<p align="center">
  Built using Python, HuggingFace Transformers & Streamlit
</p>
