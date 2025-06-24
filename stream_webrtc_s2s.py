
import os
import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pydub import AudioSegment
import queue, pydub, tempfile, time
import openai
from translator import Translator
from speech_synthesis import SpeechSynthesizer
from transcriber import Transcriber

openai.api_key = os.environ.get("OPENAI_API_KEY")

@st.cache_resource
def load_transcriber():
    """
    Load and cache the transcriber.
    """
    return Transcriber(language="en", task="transcribe")

@st.cache_resource
def load_translator():
    """
    Load and cache the translator model.
    """
    return Translator(source_lang="en", target_lang="ko")

@st.cache_resource
def load_speech_synthesizer():
    """
    Load and cache the speech synthesizer.
    """
    return SpeechSynthesizer()

def update_ui(text_container, translation_container):
    """
    Update the UI with the current transcriptions and translations.
    
    Args:
        text_container: The Streamlit container for showing the transcribed text.
        translation_container: The Streamlit container for showing the translated text.
    """
    # Update the transcription display
    with text_container.empty():
        for t in st.session_state.transcriptions:
            st.markdown(f"- {t}")
    
    # Update the translation display
    with translation_container.empty():
        for t in st.session_state.translations:
            st.markdown(f"- {t}")

def handle_silence(transcriber, translator, synthesizer, sound_chunk, silence_frames, silence_frames_threshold, text_container, translation_container, enable_speech=True):
    """
    Handle silence in the audio stream.

    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        silence_frames (int): The current number of silence frames.
        silence_frames_threshold (int): The silence frames threshold.
        text_container: The Streamlit container for showing the transcribed text.
        translation_container: The Streamlit container for showing the translated text.
        enable_speech (bool): Whether to enable speech synthesis.

    Returns:
        tuple[AudioSegment, int]: The updated sound chunk and number of silence frames.
    """
    if silence_frames >= silence_frames_threshold:
        if len(sound_chunk) > 0:
            text = transcriber.transcribe(sound_chunk)
            
            # Translate the text
            translated_text = translator.translate(text)
            
            # Only append to history if there's actual content
            if text.strip():
                # Add timestamp
                timestamp = time.strftime("%H:%M:%S")
                
                # Store the current transcription and translation
                st.session_state.current_transcription = text
                st.session_state.current_translation = translated_text
                
                # Append to session state lists
                st.session_state.transcriptions.append(f"[{timestamp}] {text}")
                st.session_state.translations.append(f"[{timestamp}] {translated_text}")
                
                # Update the UI
                update_ui(text_container, translation_container)
                
                # Synthesize and play the translated text if enabled
                if enable_speech and translated_text.strip():
                    audio = synthesizer.synthesize_speech(translated_text, lang="ko")
                    synthesizer.play_audio(audio)
            
            sound_chunk = pydub.AudioSegment.empty()
            silence_frames = 0

    return sound_chunk, silence_frames

def handle_queue_empty(transcriber, translator, synthesizer, sound_chunk, text_container, translation_container, enable_speech=True):
    """
    Handle the case where the audio frame queue is empty.

    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        text_container: The Streamlit container for showing the transcribed text.
        translation_container: The Streamlit container for showing the translated text.
        enable_speech (bool): Whether to enable speech synthesis.

    Returns:
        AudioSegment: The updated sound chunk.
    """
    if len(sound_chunk) > 0:
        # Process the same way as in handle_silence
        text = transcriber.transcribe(sound_chunk)
        
        # Translate the text
        translated_text = translator.translate(text)
        
        # Only append to history if there's actual content
        if text.strip():
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            
            # Store the current transcription and translation
            st.session_state.current_transcription = text
            st.session_state.current_translation = translated_text
            
            # Append to session state lists
            st.session_state.transcriptions.append(f"[{timestamp}] {text}")
            st.session_state.translations.append(f"[{timestamp}] {translated_text}")
            
            # Update the UI
            update_ui(text_container, translation_container)
            
            # Synthesize and play the translated text if enabled
            if enable_speech and translated_text.strip():
                audio = synthesizer.synthesize_speech(translated_text, lang="ko")
                synthesizer.play_audio(audio)
        
        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk

def display_transcriptions(text_container, translation_container):
    """
    Display the transcriptions and translations in the UI.
    
    Args:
        text_container: The Streamlit container for showing the transcribed text.
        translation_container: The Streamlit container for showing the translated text.
    """
    # Display transcriptions
    with text_container:
        for t in st.session_state.transcriptions:
            st.markdown(f"- {t}")
    
    # Display translations
    with translation_container:
        for t in st.session_state.translations:
            st.markdown(f"- {t}")

def create_download_content(transcriptions_list, include_timestamps=True):
    """
    Create formatted content for download from a list of transcriptions or translations.
    
    Args:
        transcriptions_list: List of transcriptions or translations with timestamps.
        include_timestamps: Whether to include timestamps in the output.
        
    Returns:
        str: Formatted text content for download.
    """
    if include_timestamps:
        return "\n".join(transcriptions_list)
    else:
        # Remove timestamps if requested
        cleaned_list = []
        for item in transcriptions_list:
            # Extract text after timestamp (format: "[HH:MM:SS] text")
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
    """
    The main application function for real-time speech-to-text and translation. 

    This function creates a WebRTC streamer, starts receiving audio data, processes the audio frames, 
    and transcribes the audio into text when there is silence longer than a certain threshold. It also translates the text
    and optionally synthesizes speech from the translated text.

    Args:
        status_indicator: A Streamlit object for showing the status (running or stopping).
        text_container: The Streamlit container for showing the transcribed text.
        translation_container: The Streamlit container for showing the translated text.
        download_container: The Streamlit container for download buttons.
        enable_speech (bool, optional): Whether to enable speech synthesis. Default is True.
        timeout (int, optional): Timeout for getting frames from the audio receiver. Default is 3 seconds.
        energy_threshold (int, optional): The energy threshold below which a frame is considered silence. Default is 2000.
        silence_frames_threshold (int, optional): The number of consecutive silence frames to trigger transcription. Default is 20 frames.
    """
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
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
    
    # Store WebRTC state in session state for access outside this function
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
                # Process the same way as in handle_silence
                text = transcriber.transcribe(sound_chunk)
                
                # Translate the text
                translated_text = translator.translate(text)
                
                # Only append to history if there's actual content
                if text.strip():
                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    
                    # Store the current transcription and translation
                    st.session_state.current_transcription = text
                    st.session_state.current_translation = translated_text
                    
                    # Append to session state lists
                    st.session_state.transcriptions.append(f"[{timestamp}] {text}")
                    st.session_state.translations.append(f"[{timestamp}] {translated_text}")
                    
                    # Update the UI
                    update_ui(text_container, translation_container)
                    
                    # Synthesize and play the translated text if enabled
                    if enable_speech and translated_text.strip():
                        audio = synthesizer.synthesize_speech(translated_text, lang="ko")
                        synthesizer.play_audio(audio)
                        synthesizer.close()
            break
    
    # Update download section when stream is stopped and there are transcriptions
    if not webrtc_ctx.audio_receiver and st.session_state.transcriptions:
        with download_container:
            st.markdown("### Download Transcriptions")
            
            # Create download content
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
            # Add a checkbox to enable/disable speech synthesis
            enable_speech = st.checkbox("Enable Speech Synthesis", value=True)
        with col_clear:
            # Add a button to clear history
            if st.button("Clear History"):
                st.session_state.transcriptions = []
                st.session_state.translations = []
                st.session_state.current_transcription = ""
                st.session_state.current_translation = ""
                st.rerun()

    status_indicator = st.empty()
    
    # Create two columns for transcription and translation headers
    text_section, translation_section = st.columns(2)
    with text_section:
        st.markdown("### Transcribed Text (English)")
    with translation_section:
        st.markdown("### Translated Text (Korean)")

    # Create two columns for the content
    col1, col2 = st.columns(2)
    
    # Display existing transcriptions and translations
    display_transcriptions(col1, col2)
    
    # Create a container for download buttons
    download_container = st.container()
    
    # Load the cached transcriber, translator, and synthesizer
    transcriber = load_transcriber()
    translator = load_translator()
    synthesizer = load_speech_synthesizer()
    # Pass the columns directly as containers
    app_sst(transcriber, translator, synthesizer, status_indicator, col1, col2, download_container, enable_speech=enable_speech)

if __name__ == "__main__":
    main()
