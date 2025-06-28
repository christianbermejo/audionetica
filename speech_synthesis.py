from gtts import gTTS
from pydub import AudioSegment
import pyaudio
import wave
import io
import tempfile
import threading
import queue
import re

def segment_text(text, max_segment_length=100):
    """
    Break text into segments at sentence boundaries or by length.
    """
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

import os

class SpeechSynthesizer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.is_playing = False
        self.playback_thread = None
        self._temp_files = []  # Track temporary files

    def __del__(self):
        self.close()

    def synthesize_speech(self, text, lang="ko"):
        """Convert text to speech using gTTS and return a pydub AudioSegment"""
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio = AudioSegment.from_mp3(mp3_fp)
        return audio

    def synthesize_speech_segment(self, text, lang="ko"):
        """Convert a text segment to speech using gTTS"""
        return self.synthesize_speech(text, lang)

    def synthesize_speech_streaming(self, text, lang="ko"):
        """Break text into segments and synthesize/play each segment as soon as ready"""
        segments = segment_text(text)
        for segment in segments:
            audio = self.synthesize_speech_segment(segment, lang)
            self.audio_queue.put(audio)
        # Start playback if not already playing
        if not self.is_playing:
            self.start_playback()

    def start_playback(self):
        """Start a thread to play audio segments as they become available"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.is_playing = True
            self.playback_thread = threading.Thread(target=self._playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()

    def _playback_worker(self):
        """Worker thread that plays audio segments from the queue"""
        while True:
            try:
                audio = self.audio_queue.get(timeout=5)  # Wait up to 5 seconds for new audio
                self.play_audio(audio)
                self.audio_queue.task_done()
            except queue.Empty:
                self.is_playing = False
                break

    def play_audio(self, audio):
        """Play a pydub AudioSegment through speakers using PyAudio"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self._temp_files.append(temp_file.name)
        try:
            audio.export(temp_file.name, format="wav")
            wf = wave.open(temp_file.name, 'rb')
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
            wf.close()
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                self._temp_files.remove(temp_file.name)

    def close(self):
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
            self.p = None
        # Clean up any remaining temp files
        for temp_file in self._temp_files[:]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    self._temp_files.remove(temp_file)
                except (OSError, IOError):
                    pass
