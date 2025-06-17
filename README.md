
# Audionetica 🎙️

Audionetica is a speech-to-speech translation system designed for multilingual communication across digital platforms. Its goal is to capture audio from input sources (microphone, virtual audio), transcribes to the target language and generates translate speech.

Audionetica aims to be compatible with live streaming tools like OBS.


## Tech Stack
-  **Language:** Python 3.10.X

-  **Interface:** [Streamlit](https://streamlit.io/) with [`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc) for audio input

-  **Speech Recognition:** [Whisper](https://github.com/openai/whisper) via HuggingFace `transformers`

-  **Translation:** Currently using the HuggingFace model [`seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko`](https://huggingface.co/seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko)

-  **Speech Synthesis:** [`gTTS`](https://github.com/pndurette/gTTS) and [PyAudio](https://pypi.org/project/PyAudio/)

## Dependencies

Runtime dependencies are listed in `requirements.txt`.

`pydub` requires `ffmpeg` installed on your system.

`pyaudio` requires `portaudio` installed on your system.

You can install `ffmpeg` and `portaudio` using your preferred package manager, by downloading the official installers, or by building them from source.

## Installation

1. Ensure Python 3.10 is installed.

2. Install the required Python packages:

```bash

pip  install  -r  requirements.txt

```

## Running the App

Launch the Streamlit application:

```bash

streamlit run stream_webrtc_s2s.py

```
This starts a local Streamlit server on `localhost` and immediately opens it on your browser. On the web app, allow microphone access and start speaking in English. The app will display the transcription and the Korean translation. If speech synthesis is enabled, the translated Korean text will be spoken aloud.

## Files

-  `stream_webrtc_s2s.py` - Main Streamlit application

-  `transcriber.py` - Wrapper around Whisper for transcription

-  `translator.py` - HuggingFace translation helper

-  `speech_synthesis.py` - gTTS based speech generation and playback

## License

This project does not specify a license. Use at your own discretion.