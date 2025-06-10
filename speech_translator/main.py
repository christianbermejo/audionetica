import time
import numpy as np
from speech_translator.audio_capture import AudioCapture
from speech_translator.speech_recognition import SpeechRecognizer
from speech_translator.translator import Translator
from speech_translator.speech_synthesis import SpeechSynthesizer

def is_speech(audio_data, threshold=500):
    """Simple voice activity detection"""
    return np.abs(audio_data).mean() > threshold

def main():
    # Initialize components
    audio_capture = AudioCapture()
    recognizer = SpeechRecognizer()
    translator = Translator(source_lang="en", target_lang="ko")  # English to Korean
    synthesizer = SpeechSynthesizer()
    
    print("Starting speech-to-speech translation...")
    print("Speak into your microphone. Press Ctrl+C to exit.")
    
    # Start audio stream
    audio_capture.start_stream()
    
    # Variables for state tracking
    recording = False
    silence_frames = 0
    
    try:
        while True:
            # Read audio chunk
            audio_capture.read_audio()
            
            # Get audio data for analysis
            audio_data = audio_capture.get_audio_data()
            
            if audio_data is not None:
                # Check if speech is detected
                if is_speech(audio_data) and not recording:
                    print("Speech detected, recording...")
                    recording = True
                    silence_frames = 0
                
                # If we're recording, check for end of speech
                elif recording:
                    if not is_speech(audio_data):
                        silence_frames += 1
                        
                        # If silence for ~1.5 seconds, process the speech
                        if silence_frames > 23:  # Adjusted for 1.5 seconds at 16kHz, 1024 chunk
                            print("Processing speech...")
                            
                            # Transcribe speech
                            transcription = recognizer.transcribe(audio_data)
                            print(f"Recognized: {transcription}")
                            
                            if transcription.strip():
                                # Translate text
                                translation = translator.translate(transcription)
                                print(f"Translated: {translation}")
                                
                                # Convert to speech and play
                                audio = synthesizer.synthesize_speech(translation)
                                synthesizer.play_audio(audio)
                            
                            # Reset for next utterance
                            recording = False
                            audio_capture.clear_buffer()
            
            time.sleep(0.01)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        audio_capture.close()
        synthesizer.close()

if __name__ == "__main__":
    main()
