import os
import tempfile
import queue
import threading
import time
from pydub import AudioSegment
import openai
import numpy as np

class RealtimeTranscriber:
    def __init__(self, language="en"):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI()
        self.language = language
        self.audio_buffer = queue.Queue()
        self.transcriptions = []
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_audio)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def add_audio_chunk(self, audio_chunk: AudioSegment):
        self.audio_buffer.put(audio_chunk)

    def _process_audio(self):
        """
        Process audio chunks from the buffer and transcribe using OpenAI's realtime API.
        """
        while not self._stop_event.is_set() or not self.audio_buffer.empty():
            try:
                audio_chunk = self.audio_buffer.get(timeout=0.5)
            except queue.Empty:
                continue

            text = self.transcribe_chunk(audio_chunk)
            if text.strip():
                self.transcriptions.append(text)

    def transcribe_chunk(self, audio_chunk: AudioSegment) -> str:
        """
        Transcribe an audio chunk using OpenAI's realtime transcription API.
        """
        audio_chunk = audio_chunk.set_frame_rate(16000).set_channels(1)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            audio_chunk.export(f.name, format="wav")
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
        sound = AudioSegment(
            data=audio_frame.to_ndarray().tobytes(),
            sample_width=audio_frame.format.bytes,
            frame_rate=audio_frame.sample_rate,
            channels=len(audio_frame.layout.channels),
        )
        sound_chunk += sound
        return sound_chunk
