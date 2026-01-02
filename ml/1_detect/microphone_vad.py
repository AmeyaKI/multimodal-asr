import sounddevice as sd
import numpy as np
from queue import Queue
import torch
import scipy.io.wavfile as sci_wav
from silero_vad import load_silero_vad, read_audio
from pathlib import Path
# conver to c lasses later on

OUTPUT_PATH = '../5_outputs'

SAMPLE_RATE = 16000
FRAME_SIZE = 512 
FRAME_DURATION = FRAME_SIZE / SAMPLE_RATE

# auto-end recording via silence
MAX_SILENCE = 3.0 # stop recording after 3 secs
MAX_SILENT_FRAMES = int(MAX_SILENCE / FRAME_DURATION)

VAD_THRESHOLD = 0.5


# start live microphone an+d recording
def start_microphone(audio_queue: Queue, vad_model, recorded_frames: list):
    print("Recording! Start speaking.")
    
    def audio_callback(indata, frames, time, status):
        frame = indata.copy().squeeze()
        if len(frame) < 160:
            return
        audio_queue.put(frame)

    silencio_counter = 0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                        blocksize=FRAME_SIZE, dtype="float32", 
                        callback=audio_callback):
        try:
            while True:
                frame = audio_queue.get()
                
                audio_tensor = torch.from_numpy(frame).float() # convert audio to tensor

                with torch.no_grad(): # calculate probabilities
                    speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

                if speech_prob > VAD_THRESHOLD: # if probability audio tensor is speech > 0.5 --> saved in recorded frames
                    recorded_frames.append(frame)
                    silencio_counter = 0
                else:
                    silencio_counter += 1
                    
                # stop recording if silence max reached: 3 sec
                if silencio_counter >= MAX_SILENT_FRAMES:
                    print(f"{MAX_SILENCE} seconds of silence reached. Recording stopping.")
                    break

        except KeyboardInterrupt: # stop recording manually 
            print("\nStopped recording.")
    return recorded_frames


# save recordings to .wav file
def save_recordings(recorded_frames: list):
    if recorded_frames:
        audio = np.concatenate(recorded_frames)
        audio_int16 = (audio * 32767).astype(np.int16)
        file_num = str(len(list(Path(OUTPUT_PATH).iterdir()))).zfill(5)
        
        sci_wav.write(f"{OUTPUT_PATH}/file{file_num}.wav", SAMPLE_RATE, audio_int16)
        print(f"Saved speech audio to {OUTPUT_PATH}")
    else:
        print("No speech detected. No files saved")


def main():
    vad_model = load_silero_vad()
    audio_queue = Queue()
    recorded_frames = []
    
    recorded_frames = start_microphone(audio_queue, vad_model, recorded_frames)
    save_recordings(recorded_frames)
    
if __name__ == '__main__':
    main()