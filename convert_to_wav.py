import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

def convert_wma_to_wav(wma_file, output_dir):
    filename = os.path.basename(wma_file)
    base_name = os.path.splitext(filename)[0]
    wav_file = os.path.join(output_dir, f"{base_name}.wav")
    
    if os.path.exists(wav_file):
        print(f"Skipping {filename}, already converted.")
        return

    print(f"Converting {filename}...")
    try:
        # Use ffmpeg to convert wma to wav
        # -y: overwrite output files
        # -i: input file
        subprocess.run(['ffmpeg', '-y', '-i', wma_file, wav_file], 
                       check=True, 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
        print(f"Successfully converted {filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {filename}: {e}")

def main():
    raw_dir = "raw"
    output_dir = "rawwav"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    wma_files = glob.glob(os.path.join(raw_dir, "*.wma"))
    print(f"Found {len(wma_files)} .wma files.")
    
    # Use ThreadPoolExecutor to speed up conversion if there are many files
    # Note: ffmpeg itself is multi-threaded, but converting multiple files in parallel can be faster
    with ThreadPoolExecutor(max_workers=4) as executor:
        for wma_file in wma_files:
            executor.submit(convert_wma_to_wav, wma_file, output_dir)

if __name__ == "__main__":
    main()
