#!/usr/bin/env python3
"""
YOLO Annotation Visualizer

This script visualizes YOLO format annotations to verify conversion quality.
It loads images and their corresponding label files from the converted dataset
and displays them with bounding boxes and class labels.
"""

import os
import argparse
import cv2
import numpy as np
import random
from pathlib import Path

# Class names for visualization
CLASS_NAMES = [
    'pedestrian', 'person_on_vehicle', 'car', 'bicycle', 'motorbike',
    'non_motorized_vehicle', 'static_person', 'distractor', 'occluder',
    'occluder_on_ground', 'occluder_full', 'reflection'
]

# Colors for different classes (BGR format)
COLORS = [
    (0, 0, 255),     # Red - pedestrian
    (0, 255, 0),     # Green - person on vehicle
    (255, 0, 0),     # Blue - car
    (0, 255, 255),   # Yellow - bicycle
    (255, 0, 255),   # Magenta - motorbike
    (255, 255, 0),   # Cyan - non motorized vehicle
    (128, 0, 255),   # Purple - static person
    (0, 128, 255),   # Orange - distractor
    (128, 128, 0),   # Teal - occluder
    (0, 128, 128),   # Brown - occluder on ground
    (128, 0, 128),   # Dark Purple - occluder full
    (64, 64, 255)    # Pink - reflection
]

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize YOLO annotations')
    parser.add_argument('--dataset_dir', type=str, default='../data/yolo_mot_dataset', 
                        help='Path to YOLO dataset directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of random samples to visualize')
    parser.add_argument('--output_dir', type=str, default='../output/visualization',
                        help='Directory to save visualization results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save visualization images instead of displaying them')
    return parser.parse_args()

def visualize_annotation(image_path, label_path, output_path=None, single_class=False):
    """Visualize a single image with its YOLO annotations"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    height, width, _ = image.shape
    
    # Read labels
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Label file not found {label_path}")
        return
    
    # Draw bounding boxes
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height
        
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # Choose color based on class
        if single_class:
            color = COLORS[0]  # Red for all classes
            class_name = "person"
        else:
            color = COLORS[class_id % len(COLORS)]
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw class label
        label = f"{class_name}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display or save image
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"Saved visualization to {output_path}")
    else:
        cv2.imshow('Visualization', image)
        cv2.waitKey(0)
    
    return image

def main():
    args = parse_args()
    
    # Paths
    images_dir = os.path.join(args.dataset_dir, 'images', args.split)
    labels_dir = os.path.join(args.dataset_dir, 'labels', args.split)
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return
    
    # Determine if dataset uses multiple classes or single class
    # Read data.yaml to check
    yaml_path = os.path.join(args.dataset_dir, 'data.yaml')
    single_class = True
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            for line in f:
                if line.startswith('nc:'):
                    num_classes = int(line.split(':')[1].strip())
                    single_class = (num_classes == 1)
                    break
    
    # Sample random images if needed
    if args.num_samples < len(image_files):
        sampled_images = random.sample(image_files, args.num_samples)
    else:
        sampled_images = image_files
    
    print(f"Visualizing {len(sampled_images)} images...")
    
    # Process each image
    for img_file in sampled_images:
        image_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if args.save_images:
            output_path = os.path.join(args.output_dir, os.path.basename(image_path))
            visualize_annotation(image_path, label_path, output_path, single_class)
        else:
            print(f"Displaying: {os.path.basename(image_path)}")
            visualize_annotation(image_path, label_path, None, single_class)
    
    if not args.save_images:
        cv2.destroyAllWindows()
    
    print("Visualization complete.")

if __name__ == "__main__":
    main() 