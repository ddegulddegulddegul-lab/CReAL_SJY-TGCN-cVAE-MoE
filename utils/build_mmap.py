import os
import argparse
import numpy as np
from tqdm import tqdm

def build_mmap_dataset(data_dir, mmap_path):
    print(f"1. Scanning directory '{data_dir}' for total frame size...")
    total_frames = 0
    file_list = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(".txt"):
                filepath = os.path.join(root, f)
                file_list.append(filepath)
                
    if not file_list:
        print("Error: No .txt files found in the directory!")
        return
        
    print(f"Found {len(file_list)} files. Parsing lines to determine shape...")
    
    # Automatically detect feature dimension from the first file
    sample_arr = np.loadtxt(file_list[0], dtype=np.float32, max_rows=1)
    if sample_arr.ndim == 1:
        feat_dim = sample_arr.shape[0]
    else:
        feat_dim = sample_arr.shape[1]
    
    # First pass: count total frames quickly by checking file line counts
    for fp in tqdm(file_list, desc="Counting frames"): # type: ignore
        with open(fp, 'r') as file:
            frames = sum(1 for _ in file)
            total_frames += frames
            
    print(f"Total Frames: {total_frames}, Automatically Detected Feature Dim: {feat_dim}")
    
    print(f"2. Creating Memory-Mapped Array at '{mmap_path}'...")
    # Create mmap array to avoid RAM explosion
    mmap_data = np.lib.format.open_memmap(
        mmap_path, 
        mode='w+', 
        dtype=np.float32, 
        shape=(total_frames, feat_dim)
    )
    
    print("3. Writing data to mmap...")
    current_idx = 0
    for fp in tqdm(file_list, desc="Writing files"): # type: ignore
        arr = np.loadtxt(fp, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1) # Handle single-frame files safely
        frames_in_file = arr.shape[0]
        
        # Write to disk via mmap
        mmap_data[current_idx:current_idx+frames_in_file, :] = arr[:]
        current_idx += frames_in_file
        
    mmap_data.flush()
    print("Done! Mmap dataset created successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Memory-Mapped Dataset for IIW Project")
    parser.add_argument('--data_dir', type=str, default='/home/song/reserch/iiw/Extraction', 
                        help='Path to the directory containing the source .txt files.')
    parser.add_argument('--mmap_path', type=str, default='/home/song/reserch/iiw/dataset_mmap.npy', 
                        help='Output path where the .npy mmap file should be saved.')
    
    args = parser.parse_args()
    
    # Convert to absolute paths for safety
    abs_data_dir = os.path.abspath(args.data_dir)
    abs_mmap_path = os.path.abspath(args.mmap_path)
    
    build_mmap_dataset(abs_data_dir, abs_mmap_path)
