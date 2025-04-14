#!/usr/bin/env python3
"""
MOT Frame Extractor

A utility script to extract frames at specific intervals from MOT17 and MOT20 datasets.
Useful for creating smaller datasets for quick experimentation or visualization.
"""

import os
import shutil
import argparse
import configparser
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames from MOT datasets')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Path to MOT sequence directory (e.g., MOT17/train/MOT17-02-FRCNN)')
    parser.add_argument('--output_dir', type=str, default='extracted_frames', 
                        help='Output directory')
    parser.add_argument('--frame_interval', type=int, default=10, 
                        help='Extract every N frames')
    parser.add_argument('--max_frames', type=int, default=None, 
                        help='Maximum number of frames to extract (None = all)')
    return parser.parse_args()

def get_sequence_info(seq_path):
    """Parse seqinfo.ini to get sequence information"""
    config = configparser.ConfigParser()
    config.read(os.path.join(seq_path, 'seqinfo.ini'))
    
    return {
        'name': config['Sequence']['name'],
        'imDir': config['Sequence']['imDir'],
        'frameRate': int(config['Sequence']['frameRate']),
        'seqLength': int(config['Sequence']['seqLength']),
        'imWidth': int(config['Sequence']['imWidth']),
        'imHeight': int(config['Sequence']['imHeight']),
        'imExt': config['Sequence']['imExt']
    }

def extract_frames(seq_path, output_dir, frame_interval, max_frames):
    """Extract frames at specific intervals"""
    # Get sequence info
    seq_info = get_sequence_info(seq_path)
    seq_name = seq_info['name']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process frames
    img_dir = os.path.join(seq_path, seq_info['imDir'])
    frames = sorted(os.listdir(img_dir))
    
    # Determine frames to extract
    frames_to_extract = frames[::frame_interval]
    if max_frames is not None:
        frames_to_extract = frames_to_extract[:max_frames]
    
    print(f"Extracting {len(frames_to_extract)} frames from {seq_name}")
    
    # Extract frames
    for i, frame_file in enumerate(tqdm(frames_to_extract)):
        src_img = os.path.join(img_dir, frame_file)
        dst_img = os.path.join(output_dir, f"{seq_name}_{frame_file}")
        shutil.copy(src_img, dst_img)
    
    print(f"Extracted {len(frames_to_extract)} frames to {output_dir}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    extract_frames(args.input_dir, args.output_dir, args.frame_interval, args.max_frames)

if __name__ == "__main__":
    main() 