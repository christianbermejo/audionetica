import os
import time
import queue
import pydub
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pydub import AudioSegment
from transcriber import RealtimeTranscriber
from translator import Translator
from speech_synthesizer import SpeechSynthesizer

@st.cache_resource
def load_transcriber():
    return RealtimeTranscriber(language="en")

@st.cache_resource
def load_translator():
    return Translator(source_lang="en", target_lang="ko")

@st.cache_resource
def load_speech_synthesizer():
    return SpeechSynthesizer()

def update_ui(text_container, translation_container):
    with text_container.empty():
        for t in st.session_state.transcriptions:
            st.markdown(f"- {t}")

    with translation_container.empty():
        for t in st.session_state.translations:
            st.markdown(f"- {t}")

def handle_silence(transcriber, translator, synthesizer, sound_chunk, silence_frames, silence_frames_threshold, text_container, translation_container, enable_speech=True):
    if silence_frames >= silence_frames_threshold:
        if len(sound_chunk) > 0:
            text = transcriber.transcribe_chunk(sound_chunk)
            translated_text = translator.translate(text)

            if text.strip():
                timestamp = time.strftime("%H:%M:%S")
                st.session_state.current_transcription = text
                st.session_state.current_translation = translated_text
                st.session_state.transcriptions.append(f"[{timestamp}] {text}")
                st.session_state.translations.append(f"[{timestamp}] {translated_text}")
                update_ui(text_container, translation_container)

                if enable_speech and translated_text.strip():
                    audio = synthesizer.synthesize_speech(translated_text, lang="ko")
                    synthesizer.play_audio(audio)

            sound_chunk = pydub.AudioSegment.empty()
            silence_frames = 0

    return sound_chunk, silence_frames

def handle_queue_empty(transcriber, translator, synthesizer, sound_chunk, text_container, translation_container, enable_speech=True):
    if len(sound_chunk) > 0:
        text = transcriber.transcribe_chunk(sound_chunk)
        translated_text = translator.translate(text)

        if text.strip():
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.current_transcription = text
            st.session_state.current_translation = translated_text
            st.session_state.transcriptions.append(f"[{timestamp}] {text}")
            st.session_state.translations.append(f"[{timestamp}] {translated_text}")
            update_ui(text_container, translation_container)

            if enable_speech and translated_text.strip():
                audio = synthesizer.synthesize_speech(translated_text, lang="ko")
                synthesizer.play_audio(audio)

        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk

def display_transcriptions(text_container, translation_container):
    with text_container:
        for t in st.session_state.transcriptions:
            st.markdown(f"- {t}")

    with translation_container:
        for t in st.session_state.translations:
            st.markdown(f"- {t}")

def create_download_content(transcriptions_list, include_timestamps=True):
    if include_timestamps:
        return "\n".join(transcriptions_list)
    else:
        cleaned_list = []
        for item in transcriptions_list:
            if "] " in item:
                cleaned_list.append(item.split("] ", 1)[1])
            else:
                cleaned_list.append(item)
        return "\n".join(cleaned_list)

def app_sst(
        transcriber,
        translator,
        synthesizer,
        status_indicator,
        text_container,
        translation_container,
        download_container,
        enable_speech=True,
        timeout=3,
        energy_threshold=2000,
        silence_frames_threshold=20
        ):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-speech",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096,
        media_stream_constraints={
            "video": False,
            "audio": {
                "sampleRate": 16000,
                "channelCount": 1,
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True
            }
        },
    )

    st.session_state.webrtc_active = webrtc_ctx.audio_receiver is not None

    sound_chunk = pydub.AudioSegment.empty()
    silence_frames = 0

    while True:
        if webrtc_ctx.audio_receiver:
            status_indicator.write("Currently running. Speak into the microphone.")

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                sound_chunk = handle_queue_empty(transcriber, translator, synthesizer, sound_chunk, text_container, translation_container, enable_speech)
                continue

            sound_chunk, silence_frames = transcriber.process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold)
            sound_chunk, silence_frames = handle_silence(transcriber, translator, synthesizer, sound_chunk, silence_frames, silence_frames_threshold, text_container, translation_container, enable_speech)
        else:
            status_indicator.write("Currently stopped.")
            if len(sound_chunk) > 0:
                text = transcriber.transcribe_chunk(sound_chunk)
                translated_text = translator.translate(text)

                if text.strip():
                    timestamp = time.strftime("%H:%M:%S")
                    st.session_state.current_transcription = text
                    st.session_state.current_translation = translated_text
                    st.session_state.transcriptions.append(f"[{timestamp}] {text}")
                    st.session_state.translations.append(f"[{timestamp}] {translated_text}")
                    update_ui(text_container, translation_container)

                    if enable_speech and translated_text.strip():
                        audio = synthesizer.synthesize_speech(translated_text, lang="ko")
                        synthesizer.play_audio(audio)
                        synthesizer.close()
            break

    if not webrtc_ctx.audio_receiver and st.session_state.transcriptions:
        with download_container:
            st.markdown("### Download Transcriptions")

            transcriptions_text = create_download_content(st.session_state.transcriptions)
            translations_text = create_download_content(st.session_state.translations)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download English Transcriptions (with timestamps)",
                    transcriptions_text,
                    file_name="transcriptions_with_timestamps.txt",
                    mime="text/plain"
                )

            with col2:
                st.download_button(
                    "Download Korean Translations (with timestamps)",
                    translations_text,
                    file_name="translations_with_timestamps.txt",
                    mime="text/plain"
                )

def main():
    st.title("Real-time Speech-to-Speech Translation (English to Korean)")

    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []
    if "translations" not in st.session_state:
        st.session_state.translations = []
    if "current_transcription" not in st.session_state:
        st.session_state.current_transcription = ""
    if "current_translation" not in st.session_state:
        st.session_state.current_translation = ""

    with st.expander("Settings"):
        col_speech, col_clear = st.columns(2, vertical_alignment="center")
        with col_speech:
            enable_speech = st.checkbox("Enable Speech Synthesis", value=True)
        with col_clear:
            if st.button("Clear History"):
                st.session_state.transcriptions = []
                st.session_state.translations = []
                st.session_state.current_transcription = ""
                st.session_state.current_translation = ""
                st.rerun()

    status_indicator = st.empty()

    text_section, translation_section = st.columns(2)
    with text_section:
        st.markdown("### Transcribed Text (English)")
    with translation_section:
        st.markdown("### Translated Text (Korean)")

    col1, col2 = st.columns(2)
    display_transcriptions(col1, col2)

    download_container = st.container()

    transcriber = load_transcriber()
    translator = load_translator()
    synthesizer = load_speech_synthesizer()

    app_sst(transcriber, translator, synthesizer, status_indicator, col1, col2, download_container, enable_speech=enable_speech)

if __name__ == "__main__":
    main()
