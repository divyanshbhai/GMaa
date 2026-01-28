import asyncio
import sys
import threading
import queue

from stt import VoskSTT
from llm_engine import G_Maa_Engine
from rag_engine import RagEngine

class ConversationOrchestrator:
    def __init__(self):
        print("🚀 Initializing System...")
        
        # 1. Load RAG
        self.rag = RagEngine(data_path="data/syllabus.txt")
        
        # 2. Load STT first (needed for callbacks)
        self.stt = VoskSTT(model_path="models/vosk-model-en-in-0.5")
        
        # 3. Load LLM/TTS, passing the STT mute functions
        # This tells the engine: "When you speak, call stt.set_active(False)"
        self.engine = G_Maa_Engine(
            mute_callback=lambda: self.stt.set_active(False),
            unmute_callback=lambda: self.stt.set_active(True)
        )
        
        # Communication Queue
        self.speech_queue = queue.Queue()
        self.listening_thread = None
        self.is_running = True

    def stt_worker(self):
        """Runs in a separate thread to capture audio continuously."""
        while self.is_running:
            try:
                # Increased silence_limit to 3000ms (3 seconds)
                text = self.stt.listen(silence_limit=3000)
                if text:
                    print(f"➡️ STT Thread detected: '{text}'")
                    self.speech_queue.put(text)
            except Exception as e:
                print(f"STT Error: {e}")
                break

    async def run_conversation_loop(self):
        """Main Async Loop handling Logic and TTS."""
        # Start the audio engine
        await self.engine.start()
        
        # Start the listening thread
        self.listening_thread = threading.Thread(target=self.stt_worker, daemon=True)
        self.listening_thread.start()
        
        print("\n🤖 G-Maa is ready! Speak to interrupt or start talking.\n")

        loop = asyncio.get_event_loop()

        while self.is_running:
            try:
                user_text = await loop.run_in_executor(None, self.speech_queue.get)
            except KeyboardInterrupt:
                break

            if "stop" in user_text.lower() or "shut up" in user_text.lower():
                await self.engine.interrupt()
                continue

            # 1. Retrieve RAG Context
            context = self.rag.retrieve_context(user_text)
            self.engine.set_rag_context(context)
            
            # 2. Interrupt anything currently playing
            await self.engine.interrupt()
            
            # 3. Generate Response and Speak
            # This will trigger the mute_callback automatically
            await self.engine.generate_and_speak(user_text)

        # Cleanup
        self.is_running = False
        await self.engine.stop()

if __name__ == "__main__":
    orchestrator = ConversationOrchestrator()
    try:
        asyncio.run(orchestrator.run_conversation_loop())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")