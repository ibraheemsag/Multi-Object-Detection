# 🚶‍♂️ Multi-Object Tracking Demo (MOT20-08) | YOLOv8n

A demonstration of real-time multi-object tracking using a fine-tuned YOLOv8n model on the MOT20-08 benchmark dataset. This project showcases robust object detection and tracking performance in complex urban scenes.

---

##  Demo

[![Watch the demo](http://img.youtube.com/vi/cZ2PNm11u0A/0.jpg)](https://youtu.be/cZ2PNm11u0A)

> Click the thumbnail to view the video on YouTube.

---

##  Features

-  Fine-tuned YOLOv8n model for improved pedestrian detection
-  Trained and evaluated on the [MOT20 dataset](https://motchallenge.net/data/MOT20/)
-  Real-time tracking performance on complex urban scenes
-  Integrated ByteTrack tracking algorithms
-  Comprehensive evaluation on MOT20 test sequences

---

## 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/yolo.git
cd yolo

# Set up environment using conda
conda env create -f environment.yaml
conda activate street

# Or using pip
# pip install -r requirements.txt
```

### Dataset Preparation

1. Download the MOT datasets:
   - [MOT17 Challenge](https://motchallenge.net/data/MOT17/)
   - [MOT20 Challenge](https://motchallenge.net/data/MOT20/)

2. Extract the datasets to your data directory:
   ```
   /data/
   ├── MOT17/
   └── MOT20/
   ```

3. Convert both MOT17 and MOT20 datasets to YOLO format using the provided utility:

   ```bash
   # Convert both datasets
   python utils/mot_to_yolo.py --mot17_dir ../data/MOT17 --mot20_dir ../data/MOT20 --output_dir ../data/yolo_mot_dataset
   
   # For specific MOT17 detector type (default is FRCNN)
   python utils/mot_to_yolo.py --mot17_dir ../data/MOT17 --mot20_dir ../data/MOT20 --detector FRCNN --output_dir ../data/yolo_mot_dataset
   
   # Advanced options
   python utils/mot_to_yolo.py \
       --mot17_dir ../data/MOT17 \
       --mot20_dir ../data/MOT20 \
       --output_dir ../data/yolo_mot_dataset \
       --detector FRCNN \
       --frame_interval 4 \
       --val_ratio 0.2 \
       --min_visibility 0.1 \
       --keep_all_classes True
   ```

4. Verify the conversion with the visualization tool:
   ```bash
   python utils/visualize_annotations.py --dataset_dir ../data/yolo_mot_dataset --num_samples 5
   ```

### Running Notebooks

The main functionality for model training, evaluation, and video creation is available in the Jupyter notebooks:

- **`notebooks/yolo_run.ipynb`**: Contains code for:
  - Finetuning the YOLOv8n model on the MOT20 dataset
  - Creating demo videos with tracking visualization
  - Generating result files for the MOT20 test set

- **Pre-trained Model**: The fine-tuned model is available at `notebooks/yolov8nbest.pt`

---

##  Results

### Training Performance

![Training Results](results.png)

The graphs above show the training progress of the YOLOv8n model on the MOT20 dataset over 100 epochs. The results demonstrate a strong convergence:

#### Training Loss Metrics:
- **Box Loss**: Decreased from ~2.7 to ~1.2, indicating improved bounding box accuracy on 
- **Classification Loss**: Reduced from ~2.8 to ~0.5, showing better class discrimination
- **DFL Loss**: Decreased from ~1.9 to ~1.0, reflecting better box representation

#### Performance Metrics:
- **Precision**: Reached 92% on the validation set, meaning high accuracy in the predicted detections
- **Recall**: Increased to 88%, indicating the model successfully identifies most pedestrians
- **mAP50**: Achieved 94%, which is excellent for pedestrian detection in crowded scenes
- **mAP50-95**: Peaked at 61%, showing good performance across various IoU thresholds

The training curves show consistent improvement throughout the training process, with both training and validation losses decreasing smoothly, indicating the model is learning effectively without overfitting. The high mAP50 value of 94.5% demonstrates the model's strong capability for pedestrian detection in the challenging MOT20 dataset.

### Tracking Performance

Preliminary results show significant improvements in tracking performance compared to the baseline model. The fine-tuned model demonstrates better detection accuracy and tracking consistency in crowded urban scenes.

Full benchmark results will be available after evaluation on the MOT20 benchmark server.

---

##  Future Work

- Waiting for access to official MOT evaluation server to upload results
- Comparison between pretrained model and fine-tuned model performance



---



