import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import streamlit as st
from pydub import AudioSegment
import pydub
import time

class Transcriber:
    def __init__(self, language="English", task="transcribe"):
        self.language = language
        self.task = task
        self.model, self.processor = self.load_whisper_model_and_processor()

    def load_whisper_model_and_processor(self):
        """
        Load and cache the whisper-small model and processor from HuggingFace.
        """
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        return model, processor

    def save_audio(self, audio_segment: AudioSegment, base_filename: str) -> None:
        """
        Save an audio segment to a .wav file.
        """
        filename = f"{base_filename}_{int(time.time())}.wav"
        audio_segment.export(filename, format="wav")

    def transcribe(self, audio_segment: AudioSegment, debug: bool = False) -> str:
        """
        Transcribe an audio segment using HuggingFace's Whisper-small model.
        """
        if debug:
            self.save_audio(audio_segment, "debug_audio")

        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
        input_features = self.processor(samples, sampling_rate=16000, return_tensors="pt").input_features

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=self.language,
                task=self.task
            )
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def frame_energy(self, frame):
        """
        Compute the energy of an audio frame.
        """
        samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16).astype(np.int32)
        return np.sqrt(np.mean(samples**2))

    def process_audio_frames(self, audio_frames, sound_chunk, silence_frames, energy_threshold):
        """
        Process a list of audio frames.
        """
        for audio_frame in audio_frames:
            sound_chunk = self.add_frame_to_chunk(audio_frame, sound_chunk)
            energy = self.frame_energy(audio_frame)
            if energy < energy_threshold:
                silence_frames += 1
            else:
                silence_frames = 0
        return sound_chunk, silence_frames

    def add_frame_to_chunk(self, audio_frame, sound_chunk):
        """
        Add an audio frame to a sound chunk.
        """
        sound = pydub.AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        sound_chunk += sound
        return sound_chunk
