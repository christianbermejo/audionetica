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
        self._energy_history = []
        self._current_dynamic_threshold = 2000  # Initial default threshold
        self._smoothed_energy = None

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

    def calculate_dynamic_threshold(self, audio_frames, alpha=0.95, min_threshold=100, max_threshold=5000):
        """
        Calculate a dynamic energy threshold based on recent audio frames.

        Args:
            audio_frames: List of audio frames
            alpha: Smoothing factor (0-1), higher means slower adaptation
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value

        Returns:
            Dynamic energy threshold value
        """
        for frame in audio_frames:
            energy = self.frame_energy(frame)
            self._energy_history.append(energy)

        # Keep only recent history (last 100 frames)
        if len(self._energy_history) > 100:
            self._energy_history = self._energy_history[-100:]

        if self._energy_history:
            ambient_energy = np.percentile(self._energy_history, 30)  # 30th percentile as baseline
            new_threshold = ambient_energy * 1.5  # 50% above ambient

            # Apply smoothing
            self._current_dynamic_threshold = alpha * self._current_dynamic_threshold + (1 - alpha) * new_threshold

            # Apply limits
            self._current_dynamic_threshold = max(min_threshold, min(max_threshold, self._current_dynamic_threshold))

        return self._current_dynamic_threshold

    def zero_crossing_rate(self, frame):
        """
        Calculate the zero-crossing rate of an audio frame.

        Returns:
            Zero crossing rate (crossings per sample)
        """
        samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16).astype(np.float32)
        signs = np.sign(samples)
        sign_changes = np.sum(np.abs(np.diff(signs)) > 0)
        return sign_changes / len(samples)

    def smooth_energy(self, energy):
        """
        Apply exponential smoothing to energy values.

        Args:
            energy: Current energy value

        Returns:
            Smoothed energy value
        """
        if self._smoothed_energy is None:
            self._smoothed_energy = energy

        alpha = 0.7  # Smoothing factor
        self._smoothed_energy = alpha * self._smoothed_energy + (1 - alpha) * energy

        return self._smoothed_energy

    def is_silence(self, frame, energy_threshold, zcr_threshold=0.1):
        """
        Determine if a frame is silence using multiple features.

        Args:
            frame: Audio frame
            energy_threshold: Energy threshold for silence
            zcr_threshold: Zero-crossing rate threshold

        Returns:
            Boolean indicating if the frame is silence
        """
        energy = self.frame_energy(frame)
        smoothed_energy = self.smooth_energy(energy)
        zcr = self.zero_crossing_rate(frame)

        is_silence_energy = smoothed_energy < energy_threshold
        is_silence_zcr = zcr < zcr_threshold

        return is_silence_energy and is_silence_zcr

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
