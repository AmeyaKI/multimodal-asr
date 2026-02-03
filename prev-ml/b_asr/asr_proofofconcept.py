from src.asr_class import ASR
from pathlib import Path

asr_model = ASR()
print("Model Loaded")

waveform = asr_model.process_wav(f'{Path.cwd()}/file00000.wav')
print("Waveform calcd")
text = asr_model.transcribe(waveform)
print(f"ASR Translation: {text}")