import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nlp_engine import MedicalNLP

def main():
    print("\n🏥 AI MEDICAL SCRIBE (CLI)")
    print("--------------------------")
    
    # Initialize
    app = MedicalNLP()
    
    # Default Sample
    transcript = """
    Physician: Good morning, Ms. Janet Jones. How are you?
    Patient: I have a severe headache and nausea. I took Paracetamol but it didn't help.
    Physician: It looks like a migraine. I will prescribe Sumatriptan.
    Patient: Thank you, I hope to recover in a few days.
    """
    
    print("\n🧠 Analyzing...")
    results = app.analyze(transcript)
    
    print("\n--- 1. MEDICAL SUMMARY ---")
    print(json.dumps(results['Medical_Summary'], indent=2))
    
    print("\n--- 2. SOAP NOTE ---")
    print(json.dumps(results['SOAP_Note'], indent=2))

if __name__ == "__main__":
    main()