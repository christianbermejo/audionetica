import os
import tempfile
from pydub import AudioSegment
import pyaudio
import wave
import openai

class SpeechSynthesizer:
    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI()
        self.p = pyaudio.PyAudio()

    def synthesize_speech(self, text, lang="ko"):
        """Convert text to speech using OpenAI TTS API and return a pydub AudioSegment"""
        with tempfile.NamedTemporaryFile(suffix='.mp3') as f:
            response = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text
            )
            response.stream_to_file(f.name)
            audio = AudioSegment.from_mp3(f.name)
            return audio

    def play_audio(self, audio):
        """Play a pydub AudioSegment through speakers using PyAudio"""
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            audio.export(f.name, format="wav")
            wf = wave.open(f.name, 'rb')
            stream = self.p.open(
                format=self.p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            stream.stop_stream()
            stream.close()

    def close(self):
        self.p.terminate()
