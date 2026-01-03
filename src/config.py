import torch
import sys

class Config:
    # --- MODELs ---
    # BioBERT for medical terms
    MODEL_NER = "d4data/biomedical-ner-all"
    
    # DistilBERT for sentiment
    MODEL_SENTIMENT = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Whisper for Speech-to-Text 
    WHISPER_SIZE = "base"
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        DEVICE_ID = 0
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        DEVICE_ID = -1 
    else:
        DEVICE = "cpu"
        DEVICE_ID = -1