import os
import glob
import sys
from faster_whisper import WhisperModel

def log(msg):
    try:
        # Print safely to console
        print(msg.encode('ascii', errors='replace').decode('ascii'))
        # Write to log file with utf-8
        with open("transcribe_log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def main():
    # Model settings
    model_size = "large-v3"
    log(f"Loading {model_size} model...")
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    raw_dir = "raw"
    transcripts_dir = "transcripts"
    
    # Create transcripts directory if it doesn't exist
    os.makedirs(transcripts_dir, exist_ok=True)
    
    # Get all .wma files in the raw directory
    wma_files = glob.glob(os.path.join(raw_dir, "*.wma"))
    log(f"Found {len(wma_files)} .wma files in '{raw_dir}' directory.")
    
    for i, audio_file in enumerate(wma_files, 1):
        filename = os.path.basename(audio_file)
        base_name, _ = os.path.splitext(filename)
        output_file = os.path.join(transcripts_dir, f"{base_name}.txt")
        
        # Skip if transcript already exists
        if os.path.exists(output_file):
            log(f"[{i}/{len(wma_files)}] Skipping {filename} (transcript already exists)")
            continue
            
        log(f"[{i}/{len(wma_files)}] Transcribing {filename}...")
        try:
            # Generate transcript
            segments, info = model.transcribe(audio_file, beam_size=5, language="zh")
            
            # Write to output file
            with open(output_file, "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            
            log(f"Successfully transcribed {filename}")
        except Exception as e:
            log(f"Error transcribing {filename}: {e}")

if __name__ == "__main__":
    # Clear log file
    with open("transcribe_log.txt", "w", encoding="utf-8") as f:
        f.write("Starting transcription process...\n")
    main()
