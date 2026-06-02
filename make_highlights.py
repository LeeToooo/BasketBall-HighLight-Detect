import csv
from pathlib import Path
from scipy import signal
from moviepy import VideoFileClip, concatenate_videoclips
import numpy as np

def extract_audio_energy(video_path: str, chunk_duration: float = 1.0):
    print(f"[INFO] Extracting audio energy from {video_path}")
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        print("[WARNING] No audio track found in video. Audio score will be 0.")
        return [], 1.0
        
    energies = []
    for t in np.arange(0, clip.duration, chunk_duration):
        sub = clip.audio.subclipped(t, min(t + chunk_duration, clip.duration))
        arr = sub.to_soundarray()
        if arr is not None and len(arr) > 0:
            rms = np.sqrt(np.mean(arr**2))
            energies.append(rms)
        else:
            energies.append(0)
            
    clip.close()
    
    if not energies:
        return [], 1.0
        
    # Normalize energies
    max_energy = max(energies)
    if max_energy > 0:
        energies = [e / max_energy for e in energies]
        
    return energies, chunk_duration

def smooth_and_find_peaks(csv_path: str, video_path: str, fps: float, height: float = 0.5, distance_sec: float = 3.0, audio_weight: float = 0.0):
    scores = []
    time_secs = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score_prob"]))
            time_secs.append(float(row["time_sec"]))
            
    # Extract and fuse audio energy if audio_weight > 0
    if audio_weight > 0:
        energies, chunk_duration = extract_audio_energy(video_path)
        if energies:
            print(f"[INFO] Fusing visual scores with audio (weight={audio_weight})")
            for i in range(len(scores)):
                t = time_secs[i]
                chunk_idx = min(int(t / chunk_duration), len(energies) - 1)
                audio_score = energies[chunk_idx]
                # Fuse using multiplicative formula
                scores[i] = scores[i] * (1.0 + audio_weight * audio_score)
                
    # Convert distance in seconds to frames
    distance_frames = int(fps * distance_sec)
    
    # Peak detection
    peak_indices, _ = signal.find_peaks(scores, height=height, distance=distance_frames)
    
    peak_timestamps = [time_secs[i] for i in peak_indices]
    
    print(f"[INFO] Found {len(peak_timestamps)} highlight peaks.")
    return peak_timestamps

def build_highlight_video(video_path: str, peak_timestamps: list, output_path: str, 
                          pre_peak_sec: float = 7.0, post_peak_sec: float = 3.0,
                          progress_callback=None):
    
    if not peak_timestamps:
        print("[WARNING] No peaks found, cannot build highlight video.")
        return None
        
    print(f"[INFO] Loading video: {video_path}")
    clip = VideoFileClip(video_path)
    total_duration = clip.duration
    
    subclips = []
    
    for i, peak_time in enumerate(peak_timestamps):
        start_t = max(0, peak_time - pre_peak_sec)
        end_t = min(total_duration, peak_time + post_peak_sec)
        
        # Prevent overlapping clips if peaks are too close
        if i > 0 and start_t < subclips[-1].end:
            start_t = subclips[-1].end
            
        if start_t >= end_t:
            continue
            
        print(f"  - Extracting clip {i+1}: {start_t:.1f}s to {end_t:.1f}s")
        subclip = clip.subclipped(start_t, end_t)
        subclips.append(subclip)
        
        if progress_callback:
            progress_callback(i + 1, len(peak_timestamps))
            
    if not subclips:
        print("[WARNING] No valid subclips generated.")
        return None
        
    print("[INFO] Concatenating clips...")
    final_clip = concatenate_videoclips(subclips)
    
    print(f"[INFO] Writing final highlight video to: {output_path}")
    
    # moviepy has its own logger, we might just let it print to console
    final_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    # Close clips to free memory
    clip.close()
    for sc in subclips:
        sc.close()
    final_clip.close()
    
    print("[DONE] Highlight video saved.")
    return output_path

def run_highlight_extraction(video_path: str, csv_path: str, output_dir: str, fps: float = 25.0, height: float = 0.5, audio_weight: float = 0.0, progress_callback=None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_output = str(output_path / "final_highlight.mp4")
    
    peak_timestamps = smooth_and_find_peaks(csv_path, video_path=video_path, fps=fps, height=height, audio_weight=audio_weight)
    build_highlight_video(video_path, peak_timestamps, final_output, progress_callback=progress_callback)
    
    return final_output

if __name__ == "__main__":
    run_highlight_extraction(
        video_path="data/full_matches/Full_Match-001.mp4",
        csv_path="outputs/score_inference/frame_scores.csv",
        output_dir="outputs",
        fps=30.0
    )
