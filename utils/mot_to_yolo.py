#!/usr/bin/env python3
"""
MOT to YOLO Converter

This script converts MOT17 and MOT20 datasets to YOLOv8 format.
It handles:
- Converting bounding box format
- Filtering by confidence score (ignores entries with score of 0)
- Sampling frames for training (uses every 4th frame by default)
- Using the last 20% of frames from each sequence for validation
- Creating proper train/validation splits
- Handling multiple detector types in MOT17
"""

import os
import shutil
import argparse
import configparser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MOT dataset to YOLO format')
    parser.add_argument('--mot17_dir', type=str, default='../data/MOT17', help='Path to MOT17 dataset')
    parser.add_argument('--mot20_dir', type=str, default='../data/MOT20', help='Path to MOT20 dataset')
    parser.add_argument('--output_dir', type=str, default='../data/yolo_mot_dataset', help='Output directory')
    parser.add_argument('--detector', type=str, default='FRCNN', choices=['DPM', 'FRCNN', 'SDP'], 
                        help='Detector type for MOT17 (DPM, FRCNN, SDP)')
    parser.add_argument('--frame_interval', type=int, default=4, 
                        help='Sample every N frames for training (default: 4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                        help='Validation set ratio - last x% of frames from each sequence')
    parser.add_argument('--min_visibility', type=float, default=0.1, 
                        help='Minimum visibility threshold')
    parser.add_argument('--keep_all_classes', type=bool, default=True, 
                        help='Keep all classes from MOT dataset')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create YOLO directory structure"""
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

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

def convert_mot_to_yolo_format(gt_file, img_width, img_height, min_visibility, keep_all_classes=True):
    """Convert MOT ground truth to YOLO format"""
    frame_annotations = {}
    
    # Map of MOT class IDs to YOLO class IDs
    class_mapping = {
        1: 0,  # Pedestrian
        2: 1,  # Person on vehicle
        3: 2,  # Car
        4: 3,  # Bicycle
        5: 4,  # Motorbike
        6: 5,  # Non motorized vehicle
        7: 6,  # Static person
        8: 7,  # Distractor
        9: 8,  # Occluder
        10: 9,  # Occluder on the ground
        11: 10, # Occluder full
        12: 11  # Reflection
    }
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            confidence = float(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])
            
            # Skip if confidence is 0 (ignored in evaluation)
            if confidence == 0:
                continue
                
            # Skip if visibility below threshold
            if visibility < min_visibility:
                continue
            
            # Calculate YOLO format: <class> <x_center> <y_center> <width> <height>
            x_center = (bb_left + bb_width / 2) / img_width
            y_center = (bb_top + bb_height / 2) / img_height
            width = bb_width / img_width
            height = bb_height / img_height
            
            # Map MOT class ID to YOLO class ID
            if keep_all_classes and class_id in class_mapping:
                yolo_class = class_mapping[class_id]
            else:
                # Fall back to mapping all classes to person (class 0)
                yolo_class = 0
            
            # Ensure values are in range [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []
            
            frame_annotations[frame_id].append(f"{yolo_class} {x_center} {y_center} {width} {height}")
    
    return frame_annotations

def process_sequence(seq_path, output_dir, frame_interval, val_ratio, min_visibility, keep_all_classes):
    """Process a single MOT sequence and split into train/val based on frame numbers"""
    # Get sequence info
    seq_info = get_sequence_info(seq_path)
    seq_name = seq_info['name']
    img_width = seq_info['imWidth']
    img_height = seq_info['imHeight']
    
    print(f"Processing sequence: {seq_name}")
    
    # Parse ground truth
    gt_file = os.path.join(seq_path, 'gt', 'gt.txt')
    frame_annotations = convert_mot_to_yolo_format(gt_file, img_width, img_height, min_visibility, keep_all_classes)
    
    # Process frames
    img_dir = os.path.join(seq_path, seq_info['imDir'])
    frames = sorted(os.listdir(img_dir))
    
    # Calculate split point - frames before this go to train, after to val
    total_frames = len(frames)
    val_start_idx = math.floor(total_frames * (1 - val_ratio))
    
    print(f"  Total frames: {total_frames}")
    print(f"  Training: frames 1-{val_start_idx} (every {frame_interval}th frame)")
    print(f"  Validation: frames {val_start_idx+1}-{total_frames} (all frames)")
    
    train_count = 0
    val_count = 0
    
    for i, frame_file in enumerate(tqdm(frames)):
        frame_num = int(os.path.splitext(frame_file)[0])
        
        # Skip if no annotations for this frame
        if frame_num not in frame_annotations:
            continue
        
        # Determine if this frame goes to train or val
        if i < val_start_idx:  # Training set
            # For training, use frame_interval
            if i % frame_interval != 0:
                continue
            
            split = 'train'
            train_count += 1
        else:  # Validation set - use all frames
            split = 'val'
            val_count += 1
        
        # Copy image
        src_img = os.path.join(img_dir, frame_file)
        dst_img = os.path.join(output_dir, 'images', split, f"{seq_name}_{frame_file}")
        shutil.copy(src_img, dst_img)
        
        # Create label file
        label_file = os.path.join(output_dir, 'labels', split, f"{seq_name}_{os.path.splitext(frame_file)[0]}.txt")
        with open(label_file, 'w') as f:
            for annotation in frame_annotations[frame_num]:
                f.write(f"{annotation}\n")
    
    return train_count, val_count

def create_data_yaml(output_dir, keep_all_classes):
    """Create data.yaml configuration file"""
    if keep_all_classes:
        yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

nc: 12
names: ['pedestrian', 'person_on_vehicle', 'car', 'bicycle', 'motorbike', 
        'non_motorized_vehicle', 'static_person', 'distractor', 'occluder', 
        'occluder_on_ground', 'occluder_full', 'reflection']
"""
    else:
        yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

nc: 1
names: ['person']
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

def main():
    args = parse_args()
    
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Get all MOT17 train sequences with the specified detector
    mot17_sequences = []
    if os.path.exists(args.mot17_dir):
        for seq_name in os.listdir(os.path.join(args.mot17_dir, 'train')):
            if seq_name.endswith(args.detector):
                mot17_sequences.append(os.path.join(args.mot17_dir, 'train', seq_name))
    
    # Get all MOT20 train sequences
    mot20_sequences = []
    if os.path.exists(args.mot20_dir):
        for seq_name in os.listdir(os.path.join(args.mot20_dir, 'train')):
            if not seq_name.startswith('.'):  # Skip hidden files like .DS_Store
                mot20_sequences.append(os.path.join(args.mot20_dir, 'train', seq_name))
    
    # Combine all sequences
    all_sequences = mot17_sequences + mot20_sequences
    
    if not all_sequences:
        print("No valid sequences found. Check paths and detector type.")
        return
    
    # Process all sequences
    total_train_frames = 0
    total_val_frames = 0
    
    for seq_path in all_sequences:
        train_count, val_count = process_sequence(
            seq_path, 
            args.output_dir, 
            args.frame_interval, 
            args.val_ratio, 
            args.min_visibility, 
            args.keep_all_classes
        )
        total_train_frames += train_count
        total_val_frames += val_count
    
    # Create data.yaml
    create_data_yaml(args.output_dir, args.keep_all_classes)
    
    print(f"\nConversion complete. Dataset saved to {args.output_dir}")
    print(f"Total training frames: {total_train_frames}")
    print(f"Total validation frames: {total_val_frames}")
    print("Use the following command to train YOLOv8:")
    print(f"yolo detect train data={os.path.join(args.output_dir, 'data.yaml')} model=yolov8n.pt epochs=100")

if __name__ == "__main__":
    main() 