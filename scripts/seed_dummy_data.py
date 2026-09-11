import os
import cv2
import numpy as np
import wave
import struct

def generate_audio(path, frequency=440.0, duration=3.0, sample_rate=16000):
    num_samples = int(duration * sample_rate)
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

def generate_video(path, color, duration=3.0, fps=15):
    width, height = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    num_frames = int(duration * fps)
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = color
        # Add a moving square to ensure frames aren't identical
        x = int((i / num_frames) * width)
        cv2.rectangle(frame, (x, 50), (x+50, 100), (255, 255, 255), -1)
        out.write(frame)
    out.release()

def generate_image(path, color):
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = color
    cv2.putText(img, "ID CARD", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imwrite(path, img)

def main():
    os.makedirs('data/real_voices', exist_ok=True)
    os.makedirs('data/real_faces', exist_ok=True)
    os.makedirs('data/deepfake_faces', exist_ok=True)
    os.makedirs('data/real_docs', exist_ok=True)
    os.makedirs('data/genai_docs', exist_ok=True)
    
    print("Generating dummy real voices (sine waves)...")
    for i in range(15):
        generate_audio(f'data/real_voices/dummy_{i}.wav', frequency=440.0 + (i*10))
        
    print("Generating dummy face videos (moving squares)...")
    for i in range(15):
        # real = blue, deepfake = red
        generate_video(f'data/real_faces/dummy_real_{i}.mp4', (255, 0, 0))
        generate_video(f'data/deepfake_faces/dummy_fake_{i}.mp4', (0, 0, 255))
        
    print("Generating dummy document images...")
    for i in range(20):
        # real = green, genai = purple
        generate_image(f'data/real_docs/dummy_real_{i}.jpg', (0, 255, 0))
        generate_image(f'data/genai_docs/dummy_genai_{i}.jpg', (128, 0, 128))
        
    print("Dummy data generation complete! This unblocks the pipeline safely.")

if __name__ == '__main__':
    main()
