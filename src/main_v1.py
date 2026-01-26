from mic_vad import MicVAD
from silero_vad import load_silero_vad
from asr_class import ASR
from llm.model import query
from llm.agentic import execute_command, generate_applescript
from google import genai
from tts import TTS

import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('API_KEY')

def main(): 
    # load Voice Activity Detection (VAD) model (Silero) and Mic (Sounddevice)
    vad = load_silero_vad() # 
    micvad_model = MicVAD(vad) 
    
    # load ASR (Default: Parakeet) 
    asr_model = ASR()
    
    # load TTS (Default: Bark) *** replace
    tts_model = TTS()
    
    # load llm (Default: Gemini)
    client = genai.Client(api_key=API_KEY)
    
    # PRIMITIVE MVP
    print("Initializing assistant. Press Ctrl+C to exit.")
    try:
        while True:
            user_audio_arr = micvad_model.start_microphone() # runs microphone, saves recordings, returns file number
            
            user_speech_to_text = asr_model.process_arr(user_audio_arr) # translates audio arr to text
            
            tts_model.speak("on it")
            applescript = generate_applescript(client, user_speech_to_text)
            execute_command(applescript)
            tts_model.speak("command executed")
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Assistant terminating.")


if __name__ == '__main__':
    main()