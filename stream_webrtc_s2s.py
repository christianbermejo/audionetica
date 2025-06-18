import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from pydub import AudioSegment
import queue
import threading
import pydub
import tempfile
import os
import time
import torch
from translator import Translator
from speech_synthesis import SpeechSynthesizer
from transcriber import Transcriber

@st.cache_resource
def load_transcriber():
    return Transcriber(language="English", task="transcribe")

@st.cache_resource
def load_translator():
    return Translator(source_lang="en", target_lang="ko")

@st.cache_resource
def load_speech_synthesizer():
    return SpeechSynthesizer()

def update_ui(text_container, translation_container):
    with text_container.container():
        st.markdown("### Transcribed Text (English)")
        for t in st.session_state.transcriptions:
            st.markdown(f"- {t}")
    with translation_container.container():
        st.markdown("### Translated Text (Korean)")
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
        timeout=1, 
        energy_threshold=2000, 
        silence_frames_threshold=75
        ):
    # Queues for asynchronous processing
    transcription_queue = queue.Queue()
    translation_queue = queue.Queue()
    synthesis_queue = queue.Queue()
    result_queue = queue.Queue()  # For passing results to main thread

    # Debounce and duplicate detection state
    last_transcription_time = [0]  # Use list for mutability in nested function
    min_time_between_transcriptions = 2  # seconds
    last_transcription_text = [""]  # Use list for mutability
    last_translation_text = [""]   # For translation duplicate detection

    # Buffers for batching results
    transcription_buffer = []
    translation_buffer = []
    last_session_update_time = time.time()
    session_update_interval = 1.0  # seconds

    max_history_size = 100  # Limit session state list size

    # Worker thread for transcription
    def transcription_worker():
        while True:
            audio_chunk = transcription_queue.get()
            if audio_chunk is None:
                break
            # Only process if audio chunk is at least 1 second
            if len(audio_chunk) < 1000:
                transcription_queue.task_done()
                continue
            text = transcriber.transcribe(audio_chunk)
            # Only process meaningful content (more than 5 characters)
            if len(text.strip()) > 5:
                timestamp = time.strftime("%H:%M:%S")
                result_queue.put(("transcription", timestamp, text))
                translation_queue.put((text, timestamp))
            transcription_queue.task_done()

    # Worker thread for translation
    def translation_worker():
        while True:
            item = translation_queue.get()
            if item is None:
                break
            text, timestamp = item
            translated_text = translator.translate(text)
            if translated_text.strip():
                result_queue.put(("translation", timestamp, translated_text))
                if enable_speech:
                    synthesis_queue.put(translated_text)
            translation_queue.task_done()

    # Worker thread for speech synthesis
    def synthesis_worker():
        while True:
            translated_text = synthesis_queue.get()
            if translated_text is None:
                break
            audio = synthesizer.synthesize_speech(translated_text, lang="ko")
            synthesizer.play_audio(audio)
            synthesis_queue.task_done()

    # Start worker threads
    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    translation_thread = threading.Thread(target=translation_worker, daemon=True)
    synthesis_thread = threading.Thread(target=synthesis_worker, daemon=True)
    transcription_thread.start()
    translation_thread.start()
    synthesis_thread.start()

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        media_stream_constraints={"video": False, "audio": True},
    )

    st.session_state.webrtc_active = webrtc_ctx.audio_receiver is not None

    sound_chunk = pydub.AudioSegment.empty()
    silence_frames = 0

    last_update_time = time.time()

    while True:
        if webrtc_ctx.audio_receiver:
            status_indicator.write("Currently running. Speak into the microphone.")

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                continue

            # Process audio frames for silence detection
            for audio_frame in audio_frames:
                sound_chunk = transcriber.add_frame_to_chunk(audio_frame, sound_chunk)
                energy = transcriber.frame_energy(audio_frame)
                if energy < energy_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

            current_time = time.time()
            if silence_frames >= silence_frames_threshold:
                if len(sound_chunk) > 0 and (current_time - last_transcription_time[0]) >= min_time_between_transcriptions:
                    if len(sound_chunk) >= 1000:  # Minimum 1 second of audio
                        transcription_queue.put(sound_chunk)
                        last_transcription_time[0] = current_time
                    sound_chunk = pydub.AudioSegment.empty()
                    silence_frames = 0

            # Process results from worker threads and buffer them
            while not result_queue.empty():
                result_type, timestamp, text = result_queue.get()
                if result_type == "transcription":
                    if text != last_transcription_text[0]:
                        transcription_buffer.append((timestamp, text))
                        last_transcription_text[0] = text
                elif result_type == "translation":
                    if text != last_translation_text[0]:
                        translation_buffer.append((timestamp, text))
                        last_translation_text[0] = text

            # Update session state and UI at controlled intervals
            if (current_time - last_session_update_time >= session_update_interval) and (transcription_buffer or translation_buffer):
                # Update transcriptions
                for timestamp, text in transcription_buffer:
                    st.session_state.transcriptions.append(f"[{timestamp}] {text}")
                # Update translations
                for timestamp, text in translation_buffer:
                    st.session_state.translations.append(f"[{timestamp}] {text}")
                # Limit history size
                if len(st.session_state.transcriptions) > max_history_size:
                    st.session_state.transcriptions = st.session_state.transcriptions[-max_history_size:]
                if len(st.session_state.translations) > max_history_size:
                    st.session_state.translations = st.session_state.translations[-max_history_size:]
                # Clear buffers
                transcription_buffer.clear()
                translation_buffer.clear()
                # Update UI
                update_ui(text_container, translation_container)
                last_session_update_time = current_time

        else:
            status_indicator.write("Currently stopped.")
            if len(sound_chunk) > 0:
                transcription_queue.put(sound_chunk)
                sound_chunk = pydub.AudioSegment.empty()
            break

    # Wait for all processing to finish
    transcription_queue.join()
    translation_queue.join()
    synthesis_queue.join()

    # Clean shutdown of threads
    transcription_queue.put(None)
    translation_queue.put(None)
    synthesis_queue.put(None)
    transcription_thread.join(timeout=1)
    translation_thread.join(timeout=1)
    synthesis_thread.join(timeout=1)

    # Update download section when stream is stopped and there are transcriptions
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
    st.title("Real-time Speech-to-Text with Translation")

    # Initialize session state variables
    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []
    if "translations" not in st.session_state:
        st.session_state.translations = []
    if "current_transcription" not in st.session_state:
        st.session_state.current_transcription = ""
    if "current_translation" not in st.session_state:
        st.session_state.current_translation = ""

    with st.popover("Settings"):
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

    # Create empty containers for transcription and translation
    text_container = st.empty()
    translation_container = st.empty()
    download_container = st.container()

    transcriber = load_transcriber()
    translator = load_translator()
    synthesizer = load_speech_synthesizer()

    # Initial UI update
    update_ui(text_container, translation_container)

    app_sst(transcriber, translator, synthesizer, status_indicator, text_container, translation_container, download_container, enable_speech=enable_speech)

if __name__ == "__main__":
    main()
