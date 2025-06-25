import torch
import tempfile
import threading
import queue
import wave
import pyaudio
from pydub import AudioSegment
from TTS.api import TTS

def segment_text(text, max_segment_length=100):
    """
    Break text into segments at sentence boundaries or by length.
    """
    import re
    # Split by sentence terminators
    sentence_endings = re.compile(r'([.!?])')
    sentences = []
    start = 0
    for match in sentence_endings.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
    if start < len(text):
        sentences.append(text[start:].strip())

    # Further split long sentences
    segments = []
    for sentence in sentences:
        if len(sentence) > max_segment_length:
            # Split by comma or space if too long
            parts = re.split(r'(,|\s)', sentence)
            current = ""
            for part in parts:
                if len(current) + len(part) <= max_segment_length:
                    current += part
                else:
                    if current.strip():
                        segments.append(current.strip())
                    current = part
            if current.strip():
                segments.append(current.strip())
        else:
            if sentence:
                segments.append(sentence)
    return segments

class SpeechSynthesizer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Map language codes to Coqui TTS model names or paths
        # These models should be installed or available in the environment
        self.voice_presets = {
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
            "ko": "tts_models/multilingual/multi-dataset/xtts_v2",
        }
        self.default_voice = "tts_models/en/ljspeech/tacotron2-DDC"
        self.tts_models = {}
        self.voice_history = {}

    def get_tts_model(self, lang="en"):
        if lang in self.tts_models:
            return self.tts_models[lang]
        
        model_name = self.voice_presets.get(lang, self.default_voice)
        try:
            tts = TTS(model_name)
        except TypeError as e:
            print(f"First initialization attempt failed: {e}")
            try:
                tts = TTS(model_path=model_name)
            except Exception as e2:
                print(f"Second initialization attempt failed: {e2}")
                print("Falling back to default initialization")
                tts = TTS()
        self.tts_models[lang] = tts
        return tts

    def synthesize_speech(self, text, lang="en"):
        tts = self.get_tts_model(lang)
        # Coqui TTS can synthesize directly to a numpy array or save to file
        # We'll synthesize to a temporary wav file and load it with pydub for playback
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            tts.tts_to_file(text=text, file_path=f.name, language="ko", speaker="Daisy Studious")
            audio = AudioSegment.from_wav(f.name)
        return audio

    def synthesize_speech_segment(self, text, lang="en"):
        return self.synthesize_speech(text, lang)

    def synthesize_speech_streaming(self, text, lang="en"):
        segments = segment_text(text)
        for segment in segments:
            audio = self.synthesize_speech_segment(segment, lang)
            self.audio_queue.put(audio)
        if not self.is_playing:
            self.start_playback()

    def start_playback(self):
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()

    def _playback_worker(self):
        while True:
            try:
                audio = self.audio_queue.get(timeout=5)
                self.play_audio(audio)
                self.audio_queue.task_done()
            except queue.Empty:
                self.is_playing = False
                break

    def play_audio(self, audio):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            audio.export(f.name, format="wav")
            wf = wave.open(f.name, "rb")
            stream = self.p.open(
                format=self.p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
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
