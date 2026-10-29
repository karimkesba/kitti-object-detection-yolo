# KITTI Object Detection with YOLO11

An object detection project using the KITTI dataset and Ultralytics YOLO11n. The project explores object detection through two experiments: a simplified two-class setup and an extended eight-class setup.

## Project Overview

The project aims to detect and classify objects in road-scene images from the KITTI dataset.

Two experiments were conducted:

1. **Two-Class Detection:** Focuses on Cars and Pedestrians.
2. **Eight-Class Detection:** Extends the task to all eight KITTI classes and applies additional training stages to improve detection performance.

Both experiments were implemented using Google Colab.

## Dataset

The KITTI dataset was used for training and validation.

| Split | Images |
|---|---:|
| Training | 5,985 |
| Validation | 1,496 |

The dataset contains road-scene images with YOLO-format bounding-box annotations.

### Experiment 1: Two-Class Detection

The original annotations were filtered to retain only:

- Car
- Pedestrian

The class IDs were remapped for the YOLO dataset:

| Original KITTI ID | YOLO ID | Class |
|---:|---:|---|
| 0 | 0 | Car |
| 3 | 1 | Pedestrian |

An additional Faster R-CNN experiment was explored using a pretrained Faster R-CNN ResNet50 FPN model, with its prediction head adapted for the two target classes.

### Experiment 2: Eight-Class Detection

The second experiment used all eight KITTI classes:

| Class ID | Class Name |
|---:|---|
| 0 | Car |
| 1 | Van |
| 2 | Truck |
| 3 | Pedestrian |
| 4 | Person_sitting |
| 5 | Cyclist |
| 6 | Tram |
| 7 | Misc |

## Models and Training

### Experiment 1 — Two Classes

**Faster R-CNN**
- Architecture: Faster R-CNN ResNet50 FPN.
- Pretrained COCO weights.
- Modified classification head for two classes plus background.
- One training epoch.
- Evaluated using TorchMetrics Mean Average Precision.

**YOLO11n**
- Model: `yolo11n.pt`
- Epochs: 5
- Image size: 640
- Batch size: 8
- Confidence threshold for visualization: 0.25

### Experiment 2 — Eight Classes

The YOLO11n model was trained in three successive stages:

| Stage | Epochs | Initialization |
|---|---:|---|
| Initial training | 5 | Pretrained YOLO11n |
| Fine-tuning 1 | 10 | Previous stage's `last.pt` |
| Fine-tuning 2 | 10 | Previous stage's `last.pt` |

**Training configuration:**
- Model: YOLO11n
- Image size: 640
- Batch size: 8
- Optimizer and other training settings: Ultralytics defaults
- Device: NVIDIA GPU (Google Colab)

The additional stages loaded the previous stage's last model weights and continued training.

## Results

### Experiment 1 — Two-Class Detection

**YOLO11n — 5 Epochs**

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Car | 82.9% | 76.6% | 85.4% | 59.2% |
| Pedestrian | 59.1% | 42.2% | 49.1% | 21.7% |
| **Overall** | **71.0%** | **59.4%** | **67.2%** | **40.5%** |

**Faster R-CNN**

| Metric | Result |
|---|---:|
| mAP@0.5 | 79.79% |
| mAP@0.5:0.95 | 48.57% |

*Note: Faster R-CNN was trained for one epoch. Its results are reported separately and should not be treated as a directly controlled comparison with YOLO11n.*

### Experiment 2 — Eight-Class Detection

**Final YOLO11n Results**

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Car | 91.6% | 79.0% | 90.5% | 65.5% |
| Van | 81.1% | 62.3% | 72.1% | 52.9% |
| Truck | 94.2% | 81.2% | 89.3% | 67.2% |
| Pedestrian | 76.1% | 49.6% | 62.1% | 32.2% |
| Person_sitting | 67.3% | 47.2% | 54.1% | 26.3% |
| Cyclist | 77.1% | 55.1% | 64.1% | 33.2% |
| Tram | 82.8% | 64.1% | 80.8% | 51.4% |
| Misc | 74.6% | 54.3% | 65.1% | 41.1% |
| **Overall** | **80.6%** | **61.6%** | **72.3%** | **46.2%** |

### Training Progress

| Training Stage | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Initial — 5 epochs | 66.1% | 54.3% | 59.0% | 35.1% |
| Fine-tuning 1 — 10 epochs | 73.8% | 61.4% | 69.0% | 43.5% |
| Fine-tuning 2 — 10 epochs | 80.6% | 61.6% | 72.3% | 46.2% |

The results show progressive improvement across the three training stages.

## Prediction and Visualization

The final eight-class model was evaluated on the complete validation split.

- Validation images: 1,496
- Prediction confidence threshold: 0.25
- Bounding boxes and predicted class labels were saved on the validation images.
- The prediction images were compressed into a ZIP archive and saved to Google Drive.

**Final model checkpoint:**

`runs/detect/kitti_8classes_finetune2/weights/best.pt`

## Observations and Limitations

- The dataset contains objects at different distances and scales.
- Small and distant objects can be difficult to detect and classify because they occupy fewer pixels.
- Performance varies between classes.
- Classes with fewer annotated instances, such as Person_sitting and Tram, have less training data than Car.
- YOLO11n was selected as a lightweight model; larger models and higher image resolutions could be explored in future experiments.

## Technologies Used

- Python
- PyTorch
- TorchVision
- Ultralytics YOLO11
- TorchMetrics
- OpenCV
- Matplotlib
- Google Colab

## Repository Structure

```text
kitti-object-detection-yolo/
│
├── KITTI_Object_Detection_2_Classes.ipynb
├── KITTI_Object_Detection_8_Classes.ipynb
└── README.md
```

## Conclusion

This project explored object detection on the KITTI dataset through two different class configurations.

The first experiment introduced two-class detection and explored both Faster R-CNN and YOLO11n. The second experiment extended detection to eight classes and improved YOLO11n performance through successive fine-tuning stages.

The final eight-class model achieved **72.3% mAP@50** and **46.2% mAP@50-95** on the validation set.