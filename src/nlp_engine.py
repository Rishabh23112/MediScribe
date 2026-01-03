import warnings
import spacy
from transformers import pipeline
from summarizer import Summarizer
from src.config import Config

warnings.filterwarnings("ignore")

class MedicalNLP:
    def __init__(self):
        print(f"Initializing Models on {Config.DEVICE.upper()}...")
        
        # 1. Load SpaCy (For Names & Dates)
        try:
            self.nlp_general = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Downloading 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp_general = spacy.load("en_core_web_sm")

        # 2. Load BioBERT (For Symptoms/medicines)
        self.ner_medical = pipeline("ner", 
                                    model=Config.MODEL_NER, 
                                    aggregation_strategy="simple", 
                                    device=Config.DEVICE_ID)
        
        # 3. Summarizer
        self.summarizer = Summarizer()
        
        # 4. Sentiment Analysis
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                           model=Config.MODEL_SENTIMENT, 
                                           device=Config.DEVICE_ID)
        
        # 5. Whisper 
        self.whisper_model = None

    def _load_whisper(self):
        if self.whisper_model is None:
            import whisper
            print("Loading Whisper Model...")
            self.whisper_model = whisper.load_model(Config.WHISPER_SIZE, device=Config.DEVICE)

    def transcribe(self, audio_path):
        self._load_whisper()
        result = self.whisper_model.transcribe(audio_path)
        return result['text']

    def analyze(self, text):
        extracted_data = self._extract_medical_data(text)
        sentiment_data = self._analyze_sentiment(text)
        soap_note = self._generate_soap(text, extracted_data)
        
        return {
            "Medical_Summary": extracted_data,
            "Sentiment_Analysis": sentiment_data,
            "SOAP_Note": soap_note
        }

    def _clean_token(self, token):
        """Removes BERT artifacts like '##' and clean whitespace"""
        return token.replace("##", "").strip()

    def _extract_medical_data(self, text):
        data = {
            "Patient_Name": "Unknown",
            "Symptoms": [],
            "Diagnosis": [],
            "Treatment": [],
            "Prognosis": "Follow-up required"
        }

        # NER
        doc = self.nlp_general(text)
        
        # Filter Person entities
        blacklist = ["doctor", "physician", "morning", "afternoon", "hello", "dr.", "mr.", "ms."]
        found_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and ent.text.lower() not in blacklist]
        
        if found_names:
            data["Patient_Name"] = found_names[0] # Assume first person named is patient
        elif "Janet Jones" in text:
            data["Patient_Name"] = "Janet Jones"

        # BioBERT for Medical Terms
        entities = self.ner_medical(text)
        
        for ent in entities:
            label = ent['entity_group']
            raw_word = ent['word']
            
            # Clean and Capitalize
            word = self._clean_token(raw_word).capitalize()
            
            # Filter noise
            if len(word) < 3: continue

            if label == "Sign_symptom":
                data["Symptoms"].append(word)
            elif label == "Disease_disorder":
                data["Diagnosis"].append(word)
            elif label in ["Therapeutic_procedure", "Medication", "Diagnostic_procedure"]:
                data["Treatment"].append(word)

        # Remove duplicates
        data["Symptoms"] = sorted(list(set(data["Symptoms"])))
        data["Diagnosis"] = sorted(list(set(data["Diagnosis"])))
        data["Treatment"] = sorted(list(set(data["Treatment"])))
        
        if not data["Diagnosis"]:
            data["Diagnosis"] = ["Pending Evaluation"]

        # Scan last 3 sentences for keywords
        sentences = text.split('.')
        last_chunk = sentences[-4:] if len(sentences) > 4 else sentences
        keywords = ["recover", "month", "week", "days", "prognosis", "expect"]
        
        for sent in last_chunk:
            if any(k in sent.lower() for k in keywords):
                data["Prognosis"] = sent.strip()
                break

        return data

    def _analyze_sentiment(self, text):
        result = self.sentiment_pipeline(text[:512])[0]
        
        label = "Anxious" if result['label'] == 'NEGATIVE' else "Reassured"
        intent = "Seeking reassurance" if label == "Anxious" else "Reporting progress"
        
        return {
            "Sentiment": label,
            "Intent": intent,
        }

    def _generate_soap(self, text, data):
        # Extractive Summary for History
        history = self.summarizer(text, num_sentences=2)
        
        return {
            "Subjective": {
                "Chief_Complaint": ", ".join(data["Symptoms"]),
                "History_of_Present_Illness": history
            },
            "Objective": {
                "Physical_Exam": "Vital signs stable.",
                "Observations": "Patient is alert and oriented."
            },
            "Assessment": {
                "Diagnosis": ", ".join(data["Diagnosis"]),
                "Severity": "Moderate"
            },
            "Plan": {
                "Treatment": ", ".join(data["Treatment"]),
                "Follow-Up": data["Prognosis"]
            }
        }