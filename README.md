# GMaa: Gemini-Powered Voice Assistant

This is a voice assistant project that uses a local speech-to-text engine and a large language model to respond to user queries.

## Features

*   **Speech-to-Text:** Uses the Vosk library for offline speech recognition.
*   **LLM Engine:** Integrates with the Ollama platform to leverage powerful large language models.
*   **RAG Engine:** Implements a Retrieval-Augmented Generation engine to provide more contextually relevant answers from a given data source.

## Project Structure

```
g_maa_bot/
├───.gitignore
├───llm_engine.py         # Handles interaction with the Ollama LLM
├───main.py               # Main entry point of the application
├───rag_engine.py           # Retrieval-Augmented Generation engine
├───README.md
├───requirements.txt      # Python dependencies
├───stt.py                # Speech-to-text engine using Vosk
├───data/
│   ├───history.txt       # History of the conversation
│   └───science.txt         # Data source for the RAG engine
└───models/
    ├───kokoro-v0_19.onnx   # ONNX model for Kokoro
    ├───voices.bin
    └───vosk-model-en-in-0.5/ # Vosk speech recognition model
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/divyanshbhai/GMaa.git
    cd GMaa
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Before installing the Python dependencies, you might need to install `portaudio` which is a dependency for `pyaudio`.

    **On macOS:**
    ```bash
    brew install portaudio
    ```

    **On Debian/Ubuntu:**
    ```bash
    sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
    ```

    **On Windows:**
    You will need to install `portaudio` using an installer or by building from source.

    Once `portaudio` is installed, install the Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Vosk model:**
    The Vosk model is included in the repository, so you don't need to download it separately.

## Usage

To start the voice assistant, run the `main.py` script:

```bash
python main.py
```