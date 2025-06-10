import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechRecognizer:
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def transcribe(self, audio_data, sample_rate=16000):
        """Convert audio to text using Whisper-small"""
        # Normalize audio data to float32 in [-1, 1]
        if audio_data.dtype != "float32":
            audio_data = audio_data.astype("float32") / 32768.0

        # Whisper expects mono audio at 16kHz
        input_features = self.processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)

        # Generate token ids
        predicted_ids = self.model.generate(input_features)

        # Decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]

        return transcription.strip()
