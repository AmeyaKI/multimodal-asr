# Agentic Voice-Activated AI Assistant
(Work in Progress)

Pipeline:
1. Mic + Voice Activation Detection
    - Silero VAD Model
    - local microphone
2. ASR
    - Nvidia Parakeet (0.5b params)
3. Agent: (integrating LangChain)
    Standard TTS Response
    - Inworld TTS (replacing suno bark)
    + 
    LLM and Applescript Agentic Command Execution
    - Gemini 2.5 flash lite