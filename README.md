Deep_fake_detection:

This project implements a deepfake video detection system using a CNN–RNN hybrid architecture.
Spatial features are extracted from video frames using InceptionV3, and temporal inconsistencies are modeled using a GRU-based sequence classifier.

Deepfake videos manipulate facial expressions and motion patterns over time.
While CNNs capture spatial artifacts in individual frames, temporal modeling is essential to detect inconsistencies across frames.

This project addresses that by:
Using CNNs for frame-level feature extraction
Using RNNs (GRU) to learn temporal dependencies

Model Architecture:

Video
 ├── Frame Sampling (max 20 frames)
 │
 ├── InceptionV3 (pretrained on ImageNet)
 │      ↓
 │   2048-dim feature per frame
 │
 ├── GRU (temporal modeling)
 │
 ├── Fully Connected Layers
 │
 └── Output: REAL / FAKE

 Dataset:
Training Dataset
Kaggle Deepfake Detection Challenge (DFDC)
Labels: REAL, FAKE

Data Preprocessing
Videos are read using OpenCV
Frames are:
Center-cropped
Resized to 224 × 224
Maximum of 20 frames per video
Short videos are padded using masking


Feature Extraction
Each frame is passed through InceptionV3 (without classification head):
Input: (224, 224, 3)
Output: 2048-dim feature vector
Weights: ImageNet
Frozen during training


Temporal Modeling
Two stacked GRU layers
Masking applied to ignore padded frames
Final hidden state used for classification




