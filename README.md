# Audionetica

Audionetica is a simple speech-to-speech translation demo built with Streamlit. It captures audio from a microphone via WebRTC, transcribes it using OpenAI Whisper, translates the transcription to Korean with a Hugging Face sequence-to-sequence model and optionally plays back synthesized speech using gTTS.

## Tech Stack

- **Language:** Python 3.10
- **Interface:** [Streamlit](https://streamlit.io/) with [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) for microphone input
- **Speech Recognition:** [Whisper](https://github.com/openai/whisper) via Hugging Face `transformers`
- **Translation:** Hugging Face model `seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko`
- **Speech Synthesis:** [`gTTS`](https://github.com/pndurette/gTTS) and PyAudio

## Dependencies

Runtime dependencies are listed in `requirements.txt`:

```
pyaudio
torch==2.2.2
transformers
gtts
pydub
numpy==1.26.0
SpeechRecognition
sentencepiece
sacremoses
protobuf
streamlit_webrtc
```

Additionally, `pydub` requires `ffmpeg` installed on your system.

## Installation

1. Ensure Python 3.10 is available.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Demo

Launch the Streamlit application:

```bash
python stream_webrtc_s2s.py
```

This starts a local Streamlit server. Open the printed URL in your browser, allow microphone access and start speaking in English. The app will display the transcription and the Korean translation. If speech synthesis is enabled, the translated Korean text will be spoken aloud.

## Files

- `stream_webrtc_s2s.py` - Main Streamlit application
- `transcriber.py` - Wrapper around Whisper for transcription
- `translator.py` - Hugging Face translation helper
- `speech_synthesis.py` - gTTS based speech generation and playback

## License

This project does not specify a license. Use at your own discretion.

