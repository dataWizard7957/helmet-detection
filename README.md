# Helmet Detection System

Computer vision system that detects whether workers are wearing safety helmets in construction or industrial environments.  
The model helps automate workplace safety monitoring and reduce manual inspection effort.

---

## Problem

Ensuring helmet compliance on construction sites is critical to prevent serious injuries.  
Manual monitoring is inefficient, inconsistent, and difficult to scale across large industrial environments.

An automated image-based system can monitor helmet usage and alert supervisors when violations occur.

---

## Solution

This project builds a deep learning image classification model to detect whether a worker is wearing a helmet.

The system trains multiple models and compares performance:

- Simple CNN model
- VGG16 transfer learning model
- VGG16 + Feed Forward Neural Network
- VGG16 + Data Augmentation

The final selected model uses transfer learning with VGG16 and data augmentation to improve generalization.

---

## Dataset

- Total Images: **631**
- With Helmet: **311**
- Without Helmet: **320**
- Original Image Size: **200 × 200**
- Training Size Used: **64 × 64**

Images include variations in:

- Lighting conditions
- Worker posture
- Viewing angles
- Industrial environments

---

## Model Pipeline

```
Image Dataset
      │
      ▼
Preprocessing
(RGB conversion, resize, normalization)
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Model Training
(CNN + Transfer Learning)
      │
      ▼
Evaluation
(Accuracy, Precision, Recall, F1)
      │
      ▼
Final Model Selection
```

---

## Results

| Model | Validation Accuracy |
|------|------|
| CNN (Grayscale) | ~83% |
| VGG16 Base | ~89% |
| VGG16 + FFNN | ~98.9% |
| VGG16 + FFNN + Data Augmentation | **~97–98%** |

Final model selected: **VGG16 + FFNN + Data Augmentation**

Reasons:

- Better generalization
- Reduced overfitting risk
- More robust to real-world variations

---

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Project Structure

```bash
helmet_detection
│
├── helmet_detection.py       # Main training script
│
├── data/                     # Dataset (not included in repo)
│   ├── images_proj.npy
│   └── Labels_proj.csv
│
├── requirements.txt
│
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/helmet_detection.git
cd helmet_detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Place the dataset files inside:

```bash
data/
├── images_proj.npy
└── Labels_proj.csv
```

---

# Run Training

```bash
python helmet_detection.py
```

The script will:

1. Load dataset
2. Preprocess images
3. Train CNN and VGG16 models
4. Evaluate performance
5. Select the final model

---

# Future Improvements

- Train on larger industrial datasets
- Use higher resolution images (128 or 224)
- Deploy real-time inference with CCTV feeds
- Implement helmet detection using object detection models (YOLO)
- Build safety alert dashboard for supervisors
