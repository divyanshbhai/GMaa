import pyaudio
import vosk
import json
import os
import numpy as np

class VoskSTT:
    """
    Modular Wrapper for Vosk Speech Recognition.
    Features a software 'mute' switch to prevent feedback loops.
    """
    def __init__(self, model_path="models/vosk-model-en-in-0.5"):
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk Model not found at {model_path}")

        print(f"🧠 Loading Vosk Model from {model_path}...")
        self.model = vosk.Model(model_path)
        
        # Audio Configuration
        self.sample_rate = 16000
        self.chunk_size = 1024
        
        # Feedback Prevention Control
        self.active = True

    def set_active(self, state):
        """Enable or disable the microphone input."""
        self.active = state
        # Optional: Print for debugging, remove if too noisy
        # print(f"🎤 Microphone: {'ON' if state else 'OFF (Muted)'}")

    def listen(self, silence_limit=3000): # Default changed to 3000ms (3 seconds)
        """
        Listens to microphone until silence is detected.
        Returns the transcribed text string.
        """
        p = pyaudio.PyAudio()
        
        # Setup Stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)
        
        rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
        
        # We only print "Listening" if we are actually active
        if self.active:
            print("🎤 Listening...")
        
        text_buffer = ""
        silence_frames = 0
        max_silence_frames = (self.sample_rate / self.chunk_size) * (silence_limit / 1000)
        
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                
                # --- FEEDBACK PREVENTION ---
                if not self.active:
                    silence_frames = 0 
                    continue
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    partial_text = result.get("text", "")
                    if partial_text:
                        text_buffer += partial_text + " "
                        print(f"   👂 {partial_text}")
                    silence_frames = 0
                else:
                    # --- FIX: Stable VAD Math ---
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate mean only if buffer is not empty
                    if len(audio_chunk) > 0:
                        mean_sq = np.mean(audio_chunk**2)
                        # Ensure no negative/NaN values before sqrt
                        amplitude = np.sqrt(mean_sq if mean_sq > 0 else 0)
                    else:
                        amplitude = 0
                    
                    if amplitude < 100: 
                        silence_frames += 1
                    else:
                        silence_frames = 0
                
                if silence_frames > max_silence_frames:
                    break

        except Exception as e:
            print(f"Microphone Error: {e}")
        finally:
            final_result = json.loads(rec.FinalResult())
            final_text = final_result.get("text", "")
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        full_transcript = (text_buffer + final_text).strip()
        return full_transcript