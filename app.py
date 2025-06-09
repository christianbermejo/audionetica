import streamlit as st

import pyaudio
import torch
import numpy as np
import os, time
from datetime import datetime
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import io
from pydub import AudioSegment

import warnings
# Suppress the specific NumPy UserWarning about _ARRAY_API
# This warning can appear with newer NumPy versions interacting with PyTorch
warnings.filterwarnings("ignore", message="_*ARRAY_API not found.*", category=UserWarning, module="numpy")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Language mapping for LLM models for gTTS
# Keys are display names from the dropdown
LLM_LANG_CODES = {
    "English": "en",
    "Tagalog": "tl", # Opus-MT uses 'fil' for Filipino
    "Korean": "ko"
}

GTTS_LANG_CODES = {
    "English": "en",
    "Tagalog": "tl",  # gTTS uses 'tl' for Tagalog (often used for Filipino)
    "Korean": "ko"
}

# Mapping Whisper language names/codes to LLM codes if needed for X -> English translation
# This might need expansion based on common Whisper output languages
WHISPER_TO_LLM_SOURCE_LANG = {
    "english": "en",
    "tagalog": "tl",
    "korean": "ko",
    # Add more mappings as needed
}

# init PyAudio
p = pyaudio.PyAudio()


# Helper function to safely get index of a device
def try_get_index(device_list, device_name):
    try:
        return device_list.index(device_name)
    except ValueError:
        return 0 # Default to first item if not found or list is empty

# Function to list all input devices
def list_audio_devices():
    device_list = []
    try:
        for idx in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(idx)
            if device_info.get('maxInputChannels') > 0: # Check for input channels
                 device_list.append(device_info['name'])
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        # st.error(f"Error listing audio devices: {e}")
    return device_list

# Function to list all output audio devices
def list_output_audio_devices():
    output_device_list = []
    for idx in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(idx)
        if device_info.get('maxOutputChannels') > 0:
            output_device_list.append(device_info['name'])
    return output_device_list

# Function to cache the model and processor
@st.cache_resource         
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    model.eval()
    for p_param in model.parameters(): # Renamed p to p_param
        p_param.requires_grad = False
    return processor, model

@st.cache_resource
def load_translation_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model

def translate_text(text_to_translate, target_llm_code, source_llm_code=None):
    # If source and target are the same, or text is empty, no translation needed
    if not text_to_translate.strip() or (source_llm_code and source_llm_code == target_llm_code):
        return text_to_translate
    
    # st.info(f"Using fixed translation model: {model_name}") # Info message for clarity

    # logic to switch between models based on language code
    if source_llm_code and source_llm_code != "en" and target_llm_code == "en":
        if source_llm_code == "tl": # Filipino to English
             model_name = "openai/whisper-small"
        elif source_llm_code == "ko": # Korean to English
             model_name = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko"
        else:
            #  st.warning(f"Translation from {source_llm_code} to English not directly configured. Please transcribe in English or add specific model.")
             return text_to_translate
    elif source_llm_code == "en" and target_llm_code != "en":
        if target_llm_code == "tl": # English to Filipino
            model_name = "openai/whisper-small"
        elif target_llm_code == "ko": # English to Korean
            model_name = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko"
        else:
            #  st.warning(f"Translation from English to {target_llm_code} not directly configured. Please transcribe in English or add specific model.")
             return text_to_translate
    elif source_llm_code != "en" and target_llm_code != "en":
        # st.warning(f"Direct translation from {source_llm_code} to {target_llm_code} is not supported. Try translating to English first.")
        return text_to_translate
    else: # Source is English and target is English, or other unhandled cases
        # This case is already handled by the initial check:
        # if not text_to_translate.strip() or (source_llm_code and source_llm_code == target_llm_code):
        return text_to_translate
    
    try:
        tokenizer, model_instance = load_translation_model(model_name)
        input_ids = tokenizer.encode(text_to_translate, return_tensors="pt", truncation=True).to(device)
        translated_ids = model_instance.generate(input_ids)
        translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        # st.error(f"Error during translation with {model_name}: {e}")
        return text_to_translate
    

# Load Whisper model and processor
processor, model = load_whisper_model()

# Function to process audio and generate transcription
def transcribe_audio(audio_chunk, processor_instance, model_instance, language, task): # Renamed processor, model
    input_features = processor_instance(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model_instance.generate(input_features.to(device), language=language, task=task)
    transcription = processor_instance.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

# Initialize session state to store transcriptions
if "transcriptions" not in st.session_state:
    st.session_state["transcriptions"] = ""

# Function to append transcription to session state
def update_transcription(new_text,time_taken):
    st.session_state["transcriptions"] += f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {new_text}----{time_taken:.2f}s\n"

#Streamlit UI
st.title("Favor Conference Live Translation")


if 'mic_devices' not in st.session_state:
    st.session_state.mic_devices = list_audio_devices()

if 'selected_mic_name' not in st.session_state:
    if st.session_state.mic_devices:
        st.session_state.selected_mic_name = st.session_state.mic_devices[0]
    else:
        st.session_state.selected_mic_name = None

col1_mic, col2_mic = st.columns([0.8,0.2], vertical_alignment="bottom")
with col1_mic:
    mic_index = try_get_index(st.session_state.mic_devices, st.session_state.selected_mic_name)
    if not st.session_state.mic_devices:
        st.warning("No microphones detected. Please connect a microphone and refresh.")
        selected_mic = st.selectbox("Select Microphone", [], key="mic_select", index=0)
    else:
        selected_mic = st.selectbox("Select Microphone", st.session_state.mic_devices, key="mic_select", index=mic_index)
    st.session_state.selected_mic_name = selected_mic

with col2_mic:
    refresh_mics_button = st.button("Refresh Mics", key="refresh_mics")
    if refresh_mics_button:
        st.session_state.mic_devices = list_audio_devices()
        if st.session_state.selected_mic_name not in st.session_state.mic_devices:
            if st.session_state.mic_devices:
                st.session_state.selected_mic_name = st.session_state.mic_devices[0]
            else:
                st.session_state.selected_mic_name = None
        st.rerun()

input_device = st.session_state.selected_mic_name

if 'speaker_devices' not in st.session_state:
    st.session_state.speaker_devices = ["Default"] + list_output_audio_devices()

if 'selected_speaker_name' not in st.session_state:
    st.session_state.selected_speaker_name = "Default"

col1_speaker, col2_speaker = st.columns([0.8, 0.2], vertical_alignment="bottom")
with col1_speaker:
    speaker_index = try_get_index(st.session_state.speaker_devices, st.session_state.selected_speaker_name)
    if len(st.session_state.speaker_devices) <= 1 and st.session_state.speaker_devices[0] == "Default":
        st.warning("No actual output audio devices found beyond 'Default'. Playback will use system default.")

    selected_speaker = st.selectbox("Select Output Speaker", st.session_state.speaker_devices, key="speaker_select", index=speaker_index)
    st.session_state.selected_speaker_name = selected_speaker

with col2_speaker:
    refresh_speakers_button = st.button("Refresh Speakers", key="refresh_speakers")
    if refresh_speakers_button:
        st.session_state.speaker_devices = ["Default"] + list_output_audio_devices()
        if st.session_state.selected_speaker_name not in st.session_state.speaker_devices:
            st.session_state.selected_speaker_name = "Default"
        st.rerun()

output_device_name = st.session_state.selected_speaker_name


languages = ['English', 'Tagalog', 'Korean']
# tasks = ['transcribe', 'translate']

language = st.selectbox("Choose the language to transcribe to", options=languages)
# st.write("**When you choose 'translate', it translates the audio to English**.")
task = 'transcribe'
# task = st.selectbox("Choose the task", options=tasks)

st.subheader("Text-to-Speech Options")
st.write("It will speak according the voice of provided by gTTS for the language")
# Use the same language for the audio as the transcribed language
tts_target_lang_display = language
# tts_target_lang_display = st.selectbox("Translate & Speak in", options=["English", "Tagalog", "Korean"], index=0)
speak_button = st.button("Translate and Speak Latest Transcription")

if speak_button:
    if st.session_state.get("transcriptions"):
        all_transcriptions = st.session_state["transcriptions"].strip().split('\n')
        latest_transcription_text = ""
        if all_transcriptions:
            for i in range(len(all_transcriptions) - 1, -1, -1):
                line = all_transcriptions[i]
                if line.strip():
                    if "]" in line and "----" in line:
                        try:
                            latest_transcription_text = line.split("]", 1)[1].split("----")[0].strip()
                            break
                        except IndexError:
                            continue
                    elif "----" not in line and "]" not in line and line.strip():
                        latest_transcription_text = line.strip()
                        break

        if not latest_transcription_text:
            latest_transcription_text = "No valid transcription found to speak."

        source_lang_display_name = language
        target_llm_code = LLM_LANG_CODES.get(tts_target_lang_display)
        target_gtts_code = GTTS_LANG_CODES.get(tts_target_lang_display)
        source_llm_code = WHISPER_TO_LLM_SOURCE_LANG.get(source_lang_display_name.lower(), "en")

        final_text_to_speak = latest_transcription_text

        if source_llm_code != target_llm_code:
            with st.spinner(f"Translating from {source_lang_display_name} to {tts_target_lang_display}..."):
                final_text_to_speak = translate_text(latest_transcription_text, target_llm_code, source_llm_code)

        if final_text_to_speak:
            try:
                with st.spinner(f"Generating speech in {tts_target_lang_display}..."):
                    tts = gTTS(text=final_text_to_speak, lang=target_gtts_code, slow=False)
                    mp3_fp = io.BytesIO()
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)

                    output_device_index = None
                    use_custom_playback = False
                    current_output_device_name = st.session_state.selected_speaker_name

                    if current_output_device_name and current_output_device_name != "Default":
                        st.info(f"Attempting to play audio through: {current_output_device_name}")
                        try:
                            all_devices_info = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
                            output_device_details = next((d for d in all_devices_info if d['name'] == current_output_device_name and d['maxOutputChannels'] > 0), None)

                            if not output_device_details:
                                st.warning(f"Selected output device '{current_output_device_name}' not found or not an output device. Falling back to default.")
                            else:
                                output_device_index = output_device_details['index']
                                use_custom_playback = True
                        except Exception as e:
                            st.error(f"Error identifying output device '{current_output_device_name}': {e}. Falling back to default.")

                    if use_custom_playback and output_device_index is not None:
                        try:
                            audio_segment = AudioSegment.from_file(mp3_fp, format="mp3")

                            playback_stream = p.open(format=p.get_format_from_width(audio_segment.sample_width),
                                                     channels=audio_segment.channels,
                                                     rate=audio_segment.frame_rate,
                                                     output=True,
                                                     output_device_index=output_device_index)

                            playback_stream.write(audio_segment.raw_data)

                            playback_stream.stop_stream()
                            playback_stream.close()
                            st.success(f"Played audio through: {current_output_device_name}")
                            st.success(f"Speaking: {final_text_to_speak}")


                        except Exception as e:
                            st.error(f"Could not play audio on {current_output_device_name}: {e}. Falling back to default.")
                            mp3_fp.seek(0)
                            st.audio(mp3_fp, format='audio/mp3')
                            st.success(f"Speaking (default output): {final_text_to_speak}")
                    else:
                        if current_output_device_name and current_output_device_name != "Default":
                            st.warning("Playing through default output device.")
                        mp3_fp.seek(0)
                        st.audio(mp3_fp, format='audio/mp3')
                        st.success(f"Speaking: {final_text_to_speak}")

            except Exception as e:
                st.error(f"Error generating or playing speech: {e}")
        else:
            st.warning("No text to speak after translation.")
    else:
        st.warning("No transcriptions available to speak.")

TRANSCRIPTION_INTERVAL = st.slider("Set Transcription Interval (in seconds)", min_value=5, max_value=30, value=10)

start_button = st.button('Start Transcription')
stop_button = st.button('Stop Transcription')

if start_button:
    if not input_device:
        st.error("No microphone selected or available. Cannot start transcription.")
        st.stop()

    try:
        all_devices_info = [p.get_device_info_by_index(i) for i in range(p.get_device_count())]
        input_device_details = next((d for d in all_devices_info if d['name'] == input_device and d['maxInputChannels'] > 0), None)

        if not input_device_details:
            st.error(f"Could not find details for selected microphone: {input_device}. Please refresh and try again.")
            st.stop()
        input_device_index = input_device_details['index']
    except Exception as e:
        st.error(f"Error getting microphone details: {e}. Please ensure the microphone is connected and try refreshing.")
        st.stop()


    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = int(p.get_device_info_by_index(input_device_index)['defaultSampleRate'])
    WHISPER_RATE = 16000
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=input_device_index)

    st.write(f"Listening for audio from **{input_device}**... Speak now. :studio_microphone:")
    st.write(f"Transcription interval is **{TRANSCRIPTION_INTERVAL}s** :hourglass_flowing_sand:")

    live_text = st.empty()
    st.session_state["transcriptions"] = ""
    
    audio_frames = np.array([], dtype=np.float32)

    stop_recording = False
    while not stop_recording:
        if stop_button:
            stop_recording = True
            break

        data = stream.read(RATE * TRANSCRIPTION_INTERVAL, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * WHISPER_RATE / RATE))
        
        audio_frames = np.append(audio_frames, audio_chunk)
        
        task_start = time.time()
        transcription_text = transcribe_audio(audio_chunk, processor, model, language=language, task=task)
        time_taken = time.time() - task_start
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_transcription(transcription_text,time_taken)
        live_text.markdown(f"{st.session_state['transcriptions']}")

        file_path = "transcriptions.txt"
        with open(file_path, "a") as f:
            f.write(f"[{timestamp}] {transcription_text}\n")
        
    if stop_recording:
        st.write("Stopped listening.")
        stream.stop_stream()
        stream.close()
        # p.terminate() # Do not terminate p globally here
