import pyaudio
import numpy as np
from collections import deque

class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = deque(maxlen=160)  # ~10 seconds of audio at 16kHz

    def start_stream(self):
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def read_audio(self):
        """Read audio chunk and add to buffer"""
        if self.stream:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            self.audio_buffer.append(data)
            return data
        return None

    def get_audio_data(self):
        """Get all audio data from buffer as numpy array"""
        if not self.audio_buffer:
            return None
        data = b''.join(self.audio_buffer)
        return np.frombuffer(data, dtype=np.int16)

    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer.clear()

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
