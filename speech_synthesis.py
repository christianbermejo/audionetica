from gtts import gTTS
from pydub import AudioSegment
import pyaudio
import wave
import io
import tempfile

class SpeechSynthesizer:
    def __init__(self):
        self.p = pyaudio.PyAudio()

    def synthesize_speech(self, text, lang="ko"):
        """Convert text to speech using gTTS and return a pydub AudioSegment"""
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio = AudioSegment.from_mp3(mp3_fp)
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
