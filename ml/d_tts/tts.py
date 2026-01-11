# https://docs.inworld.ai/docs/tts/tts - try 
import sounddevice as sd
import requests, base64, os
from dotenv import load_dotenv
load_dotenv()
# Inworld AI TTS
url = "https://api.inworld.ai/tts/v1/voice"

headers = {
    "Authorization": f"Basic {os.getenv('INWORLD_API_KEY')}",
    "Content-Type": "application/json"
}

payload = {
    "text": "What a wonderful day to be a text-to-speech model!",
    "voiceId": "Ashley",
    "modelId": "inworld-tts-1"
}

response = requests.post(url, json=payload, headers=headers)
response.raise_for_status()
result = response.json()
audio_content = base64.b64decode(result['audioContent'])

with open("output.wav", "wb") as f:
    f.write(audio_content)