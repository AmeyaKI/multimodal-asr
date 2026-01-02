import sounddevice as sd
import numpy as np
from queue import Queue
import torch
import scipy.io.wavfile as sci_wav
from silero_vad import load_silero_vad, read_audio
from pathlib import Path


class MicVAD():
    SAMPLE_RATE = 16000
    FRAME_SIZE = 512 
    VAD_THRESHOLD = 0.5

    def __init__(self, vad_model, audio_queue: Queue,
                 MAX_SILENCE=3.0):
        self.vad_model = vad_model
        self.audio_queue = audio_queue
        
        self.FRAME_DURATION = self.FRAME_SIZE / self.SAMPLE_RATE

        self.recorded_frames = []
        
    
    def start_microphone(self, MAX_SILENCE=3.0):
        
        MAX_SILENT_FRAMES = int(MAX_SILENCE / self.FRAME_DURATION)
        print("Recording! Start speaking.")
        
        def audio_callback(indata, frames, time, status):
            frame = indata.copy().squeeze()
            if len(frame) < 160:
                return
            self.audio_queue.put(frame)

        silencio_counter = 0
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, 
                            blocksize=self.FRAME_SIZE, dtype="float32", 
                            callback=audio_callback):
            try:
                while True:
                    frame = self.audio_queue.get()
                    
                    audio_tensor = torch.from_numpy(frame).float() # convert audio to tensor

                    with torch.no_grad(): # calculate probabilities
                        speech_prob = self.vad_model(audio_tensor, self.SAMPLE_RATE).item()

                    if speech_prob > self.VAD_THRESHOLD: # if probability audio tensor is speech > 0.5 --> saved in recorded frames
                        self.recorded_frames.append(frame)
                        silencio_counter = 0
                    else:
                        silencio_counter += 1
                        
                    # stop recording if silence max reached: 3 sec
                    if silencio_counter >= MAX_SILENT_FRAMES:
                        print(f"{MAX_SILENCE} seconds of silence reached. Recording stopping.")
                        break

            except KeyboardInterrupt: # stop recording manually 
                print("\nStopped recording.")

    # save recordings to .wav file
    def save_recordings(self, OUTPUT_PATH='../5_outputs'):
        if self.recorded_frames:
            audio = np.concatenate(self.recorded_frames)
            audio_int16 = (audio * 32767).astype(np.int16)
            file_num = str(len(list(Path(OUTPUT_PATH).iterdir()))).zfill(5) # long formatting...
            
            sci_wav.write(f"{OUTPUT_PATH}/file{file_num}.wav", self.SAMPLE_RATE, audio_int16)
            print(f"Saved speech audio to {OUTPUT_PATH}")
        else:
            print("No speech detected. No files saved")



            
    