import sounddevice as sd
import numpy as np
from queue import Queue
import torch
import scipy.io.wavfile as sci_wav
from silero_vad import load_silero_vad, read_audio
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_DURATION = 0.03 # 30ms - common fframe size
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
VAD_THRESHOLD = 0.5

OUTPUT_PATH = '../5_outputs'

vad_model = load_silero_vad()


audio_queue = Queue()

recorded_frames = []

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

print("Recording. Ctrl+C = stop.")

# start live microphone
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                    blocksize=FRAME_SIZE, dtype="float32", callback=audio_callback):
    try:
        while True:
            frame = audio_queue.get().squeeze()

            audio_tensor = torch.from_numpy(frame)

            with torch.no_grad():
                speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD: # if probability audio tensor is speech > 0.5 --> saved in recorded frames
                recorded_frames.append(frame)

    except KeyboardInterrupt:
        print("\nStopped recording.")

# save recordings to .wav file
if recorded_frames:
    audio = np.concatenate(recorded_frames)
    audio_int16 = (audio * 32767).astype(np.int16)
    file_num = str(len(list(Path(OUTPUT_PATH).iterdir()))).zfill(5)
    
    sci_wav.write(f"{OUTPUT_PATH}/file{file_num}.wav", SAMPLE_RATE, audio_int16)
    print(f"Saved speech audio to {OUTPUT_PATH}")
else:
    print("No speech detected. No files saved")
