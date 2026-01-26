# Agentic Voice-Activated AI Assistant

Pipeline (TLDR):
1. Voice: Mic + Voice Activation Detection (VAD)
    - Silero VAD (PyTorch)
    - Local microphone
2. Speech-to-Text: Automatic Speech Recognition (ASR)
    - Nvidia Parakeet (0.5b)
3. Agent: (integrating LangChain)
    - Text-to-Speech: Inworld TTS
    - Agentic Capabilities: LLM and Applescript Agentic Command Execution
       - LLM: Gemini 2.5 flash lite (faster, free api)
