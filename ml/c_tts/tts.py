# https://docs.inworld.ai/docs/tts/tts - try 
import io
import sounddevice as sd
import soundfile as sf
import requests, base64, os
from dotenv import load_dotenv
load_dotenv()
from scipy.io.wavfile import write as wav_write

# Inworld AI TTS
class inw_TTS():
    URL = "https://api.inworld.ai/tts/v1/voice"
    HEADERS = {
        "Authorization": f"Basic {os.getenv('INWORLD_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    def __init__(self, model='inworld-tts-1'):
        self.model=model
        
    def speak(self, user_text: str, voice='Ashley'):
        payload = {
            "text": user_text,
            "voiceId": voice,
            "modelId": self.model
        }

        response = requests.post(self.URL, json=payload, headers=self.HEADERS)
        response.raise_for_status()
        result = response.json()
        audio_bytes = base64.b64decode(result['audioContent'])

        def save_or_play():
            audio_buffer = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")

            sd.play(audio_data, sample_rate)
            sd.wait()
          
        return save_or_play()