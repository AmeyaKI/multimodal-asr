from src.mic_vad import MicVAD
from silero_vad import load_silero_vad
from src.asr_class import ASR
# from c_llm.llm import query
# from c_llm.agentic import execute_command, generate_applescript
from google import genai
from tts import TTS

import langchain
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('API_KEY')

vad = load_silero_vad() # load Voice Activity Detection (VAD) model (Silero) and Mic (Sounddevice)
micvad_model = MicVAD(vad) 
asr_model = ASR() # load ASR (Default: Parakeet) 
tts_model = TTS() # load TTS (Default: Inworld)
client = genai.Client(api_key=API_KEY) # load llm (Default: Gemini)



user_audio_arr = micvad_model.start_microphone() # runs microphone, saves recordings, returns file number        
user_speech_to_text = asr_model.process_arr(user_audio_arr) # translates audio arr to text


# Implement LangChain

"""
PIPE
1. VAD/Mic (non lang)
2. ASR (output --> lang input)

LANG
3. LLM to determine command
    1. ALWAYS: TTS respond ("on it.")
    2. Specialized agents
        - Calendar Agent (special prompt sent in as instructions
        - ..
    3. Response (TTS)
        If command failed --> xxx
        command worked --> xxx
"""
