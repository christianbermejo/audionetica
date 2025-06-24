import os
import tempfile
from pydub import AudioSegment
import pydub
import time
import openai
import numpy as np

class Transcriber:
    def __init__(self, language="en", task="transcribe"):
        self.language = language
        self.task = task
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI()

    def save_audio(self, audio_segment: AudioSegment, base_filename: str) -> None:
        """
        Save an audio segment to a .wav file.
        """
        filename = f"{base_filename}_{int(time.time())}.wav"
        audio_segment.export(filename, format="wav")

    def transcribe(self, audio_segment: AudioSegment, debug: bool = False) -> str:
        """
        Transcribe an audio segment using OpenAI's Realtime API.
        """
        if debug:
            self.save_audio(audio_segment, "debug_audio")

        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            audio_segment.export(f.name, format="wav")
            with open(f.name, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.language.lower()
                )
        return response.text

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
