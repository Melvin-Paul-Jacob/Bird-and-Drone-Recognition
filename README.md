# Bird and Drone Detection & Tracking from UAV Camera

# Objective
Develop a computer vision and deep learning model capable of detecting and tracking birds and drones from an onboard camera installed on a UAV. The model should effectively differentiate between birds and drones, even when they appear as small objects in the image (low pixel count). The focus is on building an efficient, explainable, and deployable solution suitable for resource-constrained environments.

# Datasets
The following are the links to the datasets used in this project:
- [Abdulrahman Eidhah](https://universe.roboflow.com/abdulrahman-eidhah/final-jdwbv)
- [Dr. Ali Hilal, Dr. Maher Al-Baghdadi](https://www.kaggle.com/datasets/hussein1234/drone-uav-bird/data)

# Data Augmentations
The following augmentations were performed randomly of the dataset (using the roboflow platform):
- Resize to 640x640 (Fit within)
- Horizontal Flip
- Varied Saturation: +-25%
- Varied Exposure: +-10%
- Varied Brightness: +-15%
- Tiling (not used due to lack of suitable dataset)

Resultant dataset had ~11000 images

# Base Model
The model used is the Yolov8 model due to the use of SPPF (Spatial Pyramid Pooling Fusion) in its architecture. The purpose of SPPF is to provide a multi-scale representation of the input feature maps. By pooling at different scales, SPPF allows the model to capture features at various levels of abstraction. This can be particularly useful in object detection, where objects of different sizes may need to be detected.  It is an optimized version with the same mathematical functionality of SPP (Spatial Pyramid Pooling) originally used in YOLOv3, but fewer floating-point operations (FLOPs). The specific architecture used in this project is yolov8n for a balance in speed and accuracy.  

# Model Optimizations
Possible optimizations are as follows:
- Pruning: 
- Static Quantization
- Model Distillation
- Layer Fusion
- SAHI (Slicing Aided Hyper Inference) (not used due to lack of suitable dataset)