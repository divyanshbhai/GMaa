
# G-Maa: A Voice-based Conversational AI

G-Maa is a voice-based conversational AI that acts like a loving grandmother. It uses a combination of offline and online services to provide a responsive and interactive experience.

## Features

- **Voice-based Interaction:** G-Maa listens to your voice and responds in a grandmotherly voice.
- **Interruptible:** You can interrupt G-Maa at any time by simply speaking.
- **RAG Integration:** G-Maa uses a Retrieval-Augmented Generation (RAG) engine to answer questions based on a provided text file.
- **Offline First:** The STT and TTS engines run completely offline.
- **Customizable:** You can easily change the LLM model, TTS voice, and RAG data.

## Project Structure

```
├── llm_engine.py           # Handles the Large Language Model (LLM) and Text-to-Speech (TTS)
├── main.py                 # The main entry point of the application
├── rag_engine.py           # Handles the Retrieval-Augmented Generation (RAG)
├── requirements.txt        # Python dependencies
├── stt.py                  # Handles Speech-to-Text (STT)
├── data/
│   ├── history.txt         # Conversation history
│   └── science.txt         # Data for the RAG engine
└── models/
    ├── kokoro-v0_19.onnx     # TTS model
    ├── voices.bin            # TTS voices
    └── vosk-model-en-in-0.5/ # STT model
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/divyanshbhai/GMaa.git
cd GMaa
```

### 2. Install dependencies

First, you need to install the required system dependencies.

**For Debian/Ubuntu:**

```bash
sudo apt-get install portaudio19-dev
```

**For macOS:**

```bash
brew install portaudio
```

Then, install the Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download the models

The STT and TTS models are not included in this repository. You need to download them separately.

- **STT Model:** Download the Vosk model for your language from the [Vosk website](https://alphacephei.com/vosk/models) and place it in the `models/` directory. The current model used is `vosk-model-en-in-0.5`.
- **TTS Model:** Download the Kokoro model from the [Kokoro website](https://github.com/mut-ex/kokoro-speech/releases) and place it in the `models/` directory. The current model used is `kokoro-v0_19.onnx`.

### 4. Run the application

To run the application, simply run the `main.py` file:

```bash
python main.py
```

## How it works

The application is divided into four main components:

- **`main.py`:** This is the main entry point of the application. It initializes the other components and runs the main conversation loop.
- **`stt.py`:** This component uses the Vosk library to convert speech to text. It runs in a separate thread and continuously listens for voice input.
- **`llm_engine.py`:** This component uses the Ollama library to generate a response from the LLM. It also uses the Kokoro library to convert the text response to speech.
- **`rag_engine.py`:** This component uses the `sentence-transformers` library to find the most relevant text chunks from a given text file.

The application works as follows:

1. The `stt.py` component listens for voice input and converts it to text.
2. The `main.py` component receives the text from the `stt.py` component.
3. The `main.py` component sends the text to the `rag_engine.py` component to get the relevant context.
4. The `main.py` component sends the text and context to the `llm_engine.py` component.
5. The `llm_engine.py` component generates a response from the LLM and converts it to speech.
6. The `llm_engine.py` component plays the speech.

The application uses a queue to communicate between the different components. The `stt.py` component puts the transcribed text into a queue, and the `main.py` component gets the text from the queue. The `llm_engine.py` component puts the generated audio into a queue, and the `main.py` component gets the audio from the queue and plays it.
