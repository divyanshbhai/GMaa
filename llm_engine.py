import asyncio
import os
import platform
import wave
import numpy as np
from kokoro_onnx import Kokoro
from ollama import AsyncClient
from typing import List
import traceback
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:0.5b"
LLM_CTX_SIZE = 1024

TTS_MODEL_PATH = "models/kokoro-v0_19.onnx"
TTS_VOICE_PATH = "models/voices.bin"
TTS_VOICE_ID = "af_heart"
TTS_SPEED = 1.1

MAX_WORDS = 18
SILENCE_TIMEOUT_MS = 800
MIN_WORDS = 2

class SmartPhraseBuffer:
    """Intelligently buffers tokens to speak in linguistically meaningful units."""
    def __init__(self, dispatch_callback):
        self.buffer = ""
        self.dispatch_callback = dispatch_callback
        self.last_token_time = 0
        self.silence_task = None

    def add_token(self, token: str):
        self.buffer += token
        self.last_token_time = asyncio.get_event_loop().time()
        if self.silence_task:
            self.silence_task.cancel()
        # Trigger faster (300ms instead of 800ms)
        self.silence_task = asyncio.create_task(self._silence_watchdog())

    async def _silence_watchdog(self):
        await asyncio.sleep(0.3) 
        if self.buffer.strip():
            self._try_dispatch(force=True)

    def _try_dispatch(self, force=False):
        text = self.buffer.strip()
        if not text: return
        words = text.split()
        word_count = len(words)

        # Minimum threshold (Avoid single letter stutters)
        if not force and word_count < 2: return
        
        # Punctuation Check (Immediate Trigger)
        # Added Comma (,) to this list to speak in natural chunks
        if text[-1] in ".!?,":
            self._do_dispatch(text)
            return
        
        conjunctions = {"and", "but", "or", "so", "because"}
        if not force and words[-1].lower() in conjunctions: return

        # Length Check (Forced Flush)
        # Reduced to 8 words for snappier response
        if word_count >= 8 or force:
            split_point = self._find_safe_split_point(words)
            if split_point > 0:
                chunk = " ".join(words[:split_point])
                remainder = " ".join(words[split_point:])
                self._do_dispatch(chunk)
                self.buffer = remainder + " "
            else:
                self._do_dispatch(text)

    def _find_safe_split_point(self, words: List[str]) -> int:
        for i in range(len(words) - 1, 0, -1):
            word = words[i]
            if word[-1] == ",": return i + 1
            if word.lower() in {"and", "but", "or"}: return i
            if i + 1 < len(words) and words[i+1][0].isupper():
                 if word.lower() in {"in", "at", "to", "from", "of", "on"}: return i
        return 0

    def _do_dispatch(self, text):
        # print(f"🔄 Buffer Dispatching: '{text}'") # Commented out for cleaner logs
        self.dispatch_callback(text)
        self.buffer = ""
        if self.silence_task:
            self.silence_task.cancel()

class G_Maa_Engine:
    def __init__(self, mute_callback=None, unmute_callback=None):
        print("👵 Initializing G-Maa Engine...")
        self.llm_client = AsyncClient(host=OLLAMA_HOST)
        
        # Concurrency Control
        self.executor = ThreadPoolExecutor(max_workers=2) 
        self.mute_callback = mute_callback
        self.unmute_callback = unmute_callback
        
        self.player_cmd = "aplay"
        if platform.system() == "Darwin": self.player_cmd = "afplay"

        try:
            self.kokoro = Kokoro(TTS_MODEL_PATH, TTS_VOICE_PATH)
            self.generate_method = self.kokoro.create
            print("✅ TTS Kokoro Initialized")
        except Exception as e:
            print(f"❌ TTS Init Failed: {e}")
            self.kokoro = None

        # RESTORE: Queue for ordered execution
        self.audio_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.playback_task = None
        self.phrase_buffer = SmartPhraseBuffer(self._queue_tts_generation)
        self.history = [] 
        self.rag_context = ""

    async def start(self):
        self.stop_event.clear()
        # We wrap the worker in a safety loop to auto-restart if it crashes
        self.playback_task = asyncio.create_task(self._playback_supervisor())

    async def _playback_supervisor(self):
        """Supervisor loop to keep playback running even if it crashes."""
        while not self.stop_event.is_set():
            print("🔂 Starting Playback Worker...")
            try:
                await self._playback_worker()
            except Exception as e:
                print(f"💥 Playback Worker Crashed: {e}")
                traceback.print_exc()
                print("🔄 Restarting in 1 second...")
                await asyncio.sleep(1)

    async def stop(self):
        self.stop_event.set()
        await self.interrupt()
        if self.playback_task:
            self.playback_task.cancel()

    async def interrupt(self):
        # Clear pending tasks from queue
        while not self.audio_queue.empty():
            try:
                task = self.audio_queue.get_nowait()
                if not task.done():
                    task.cancel()
            except asyncio.QueueEmpty:
                break
        
        # Reset Buffer
        self.phrase_buffer.buffer = ""
        self.phrase_buffer._do_dispatch("") 
        
        # Unmute mic
        if self.mute_callback:
            self.unmute_callback()
        self.is_speaking = False
        
        print("🛑 Interrupted")

    async def _playback_worker(self):
        """
        Worker that manages ordered playback and intelligent muting.
        It UNMUTES the mic while waiting for generation (allowing interruption),
        and MUTES it only while actual audio is playing.
        """
        self.is_speaking = False
        
        while not self.stop_event.is_set():
            
            # 1. If Queue is completely empty -> Unmute (Safe to listen)
            if self.audio_queue.empty():
                if self.is_speaking:
                    if self.unmute_callback:
                        self.unmute_callback()
                    self.is_speaking = False
                await asyncio.sleep(0.1)
                continue

            # 2. Get next task in order
            task = self.audio_queue.get_nowait()

            # --- STRATEGIC UNMUTE ---
            # We have a task to process, but we haven't generated or played it yet.
            # If generation is slow, we should LISTEN for interrupts.
            if self.is_speaking:
                if self.unmute_callback:
                    self.unmute_callback()
                self.is_speaking = False

            # 3. Wait for generation (Concurrent generation happens here)
            # If this takes 0.5s, the mic is ON for those 0.5s.
            try:
                temp_path = await task
            except asyncio.CancelledError:
                # Task was cancelled by engine.interrupt()
                continue
            except Exception as e:
                print(f"Task Error: {e}")
                continue

            if not temp_path:
                # Task returned None (empty text) -> Keep listening
                continue

            if not os.path.exists(temp_path):
                print(f"⚠️ File not found: {temp_path}")
                continue
            
            # 4. PLAYBACK PHASE -> MUTE MIC
            try:
                size = os.path.getsize(temp_path)
                if size < 100:
                    continue
            except:
                continue

            # Mute immediately before playing to avoid echo
            if not self.is_speaking:
                if self.mute_callback:
                    self.mute_callback()
                self.is_speaking = True

            try:
                proc = await asyncio.create_subprocess_exec(
                    self.player_cmd, temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()

                # Cleanup
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except: pass

                # Note: We do NOT unmute here.
                # The loop will start over, see if the queue is empty.
                # If empty -> Unmute. If more chunks -> Unmute (while waiting).
            
            except Exception as e:
                print(f"Playback Error: {e}")
                if os.path.exists(temp_path):
                    try: os.remove(temp_path) 
                    except: pass
                self.is_speaking = False

    def _queue_tts_generation(self, text_chunk):
        # 1. Create the task (starts generating immediately in background)
        task = asyncio.create_task(self._generate_speech_file(text_chunk))
        
        # 2. Put the TASK in the queue immediately.
        # This guarantees the queue order is correct, even if B finishes generating before A.
        self.audio_queue.put_nowait(task)

    async def _generate_speech_file(self, text: str):
        if not self.kokoro: return None
        
        text = text.strip().replace('\n', ' ')
        
        if len(text) < 2 or text in [".", ",", "!", "?", ":", ";", "-", "...", "\"", "'"]:
            return None

        fname = os.path.abspath(f"temp_audio_{asyncio.get_event_loop().time()}.wav")
        
        try:
            # Run the heavy TTS calculation in a background thread
            loop = asyncio.get_event_loop()
            audio, sr = await loop.run_in_executor(self.executor, self._blocking_tts_call, text)
            
            if audio is None or len(audio) == 0:
                return None

            audio_int16 = (audio * 32767).astype(np.int16)
            
            with wave.open(fname, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
            
            # FIX: RETURN the filename. Do NOT put it in the queue here.
            return fname
            
        except Exception as e:
            print(f"❌ ERROR GENERATING AUDIO: {e}")
            return None

    def _blocking_tts_call(self, text: str):
        """The blocking CPU-intensive part. Runs in separate thread."""
        return self.generate_method(text, voice=TTS_VOICE_ID, speed=TTS_SPEED, lang='en-us')

    def set_rag_context(self, context: str):
        self.rag_context = context

    async def generate_and_speak(self, user_input: str):
        system_instruction = (
            "You are G-Maa, a loving grandmother. Use very short, kind sentences. "
            "You are talking to a child."
        )
        
        if self.rag_context:
            system_instruction += (
                f"\n\nUse the following syllabus content to answer if relevant: "
                f"{self.rag_context}"
            )

        messages = [{'role': 'system', 'content': system_instruction}]
        messages.extend(self.history) 
        messages.append({'role': 'user', 'content': user_input})

        print(f"👤 User: {user_input}")
        print(f"👵 G-Maa: ...")

        try:
            stream = await self.llm_client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options={'num_ctx': LLM_CTX_SIZE, 'temperature': 0.7}
            )

            full_response = ""
            async for chunk in stream:
                token = chunk['message']['content']
                full_response += token
                self.phrase_buffer.add_token(token)
                self.phrase_buffer._try_dispatch()

            self.phrase_buffer._try_dispatch(force=True)
            
            print(f"🤖 Full LLM Response: '{full_response}'")
            
            self.history.append({'role': 'user', 'content': user_input})
            self.history.append({'role': 'assistant', 'content': full_response})
            if len(self.history) > 8:
                self.history = self.history[-8:]

        except Exception as e:
            print(f"LLM Error: {e}")
            await self._queue_tts_generation("I'm having trouble thinking right now.")