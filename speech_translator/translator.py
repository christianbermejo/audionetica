import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, source_lang="en", target_lang="ko"):
        # Commented out: Helsinki-NLP MarianMT model
        # model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Use the Korean translation model
        model_name = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def translate(self, text):
        """Translate text from source language to target language"""
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translation
        translated_ids = self.model.generate(**inputs)

        # Decode translation
        translated_text = self.tokenizer.batch_decode(
            translated_ids, 
            skip_special_tokens=True
        )[0]

        return translated_text.strip()
