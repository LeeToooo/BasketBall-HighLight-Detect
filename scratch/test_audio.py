import os
from moviepy import VideoFileClip
import numpy as np

def extract_audio_energy(video_path):
    print("Loading video:", video_path)
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        print("No audio found.")
        return
    
    # We can use iter_frames or to_soundarray, but to_soundarray might load the whole audio into memory.
    # For a 10 min video, 44100 fps * 600 s * 2 channels * 4 bytes = 211 MB, which is fine.
    # But wait, we just want the volume per second or half-second.
    print("Getting audio array...")
    
    # Let's read in chunks to avoid memory explosion if the video is huge
    chunk_duration = 1.0 # 1 second chunks
    fps = clip.audio.fps
    
    energies = []
    
    for t in np.arange(0, clip.duration, chunk_duration):
        sub = clip.audio.subclipped(t, min(t + chunk_duration, clip.duration))
        arr = sub.to_soundarray()
        if arr is not None and len(arr) > 0:
            rms = np.sqrt(np.mean(arr**2))
            energies.append(rms)
        else:
            energies.append(0)
            
    print("Computed energies:", len(energies))
    print("Max energy:", max(energies))
    print("Min energy:", min(energies))
    
if __name__ == "__main__":
    extract_audio_energy("data/full_matches/Full_Match-001.mp4")
