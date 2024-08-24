# Bird and Drone Detection & Tracking from UAV Camera

# Objective
Develop a computer vision and deep learning model capable of detecting and tracking birds and drones from an onboard camera installed on a UAV. The model should effectively differentiate between birds and drones, even when they appear as small objects in the image (low pixel count). The focus is on building an efficient, explainable, and deployable solution suitable for resource-constrained environments.

# Datasets
The following are the links to the datasets used in this project:
- [Item 1](https://universe.roboflow.com/abdulrahman-eidhah/final-jdwbv)
- [Item 2](https://www.kaggle.com/datasets/hussein1234/drone-uav-bird/data)

# Base Model
The model used is the Yolov8 model due to the use of SPPF (Spatial Pyramid Pooling Fusion) in its archetecture. The purpose of SPPF is to provide a multi-scale representation of the input feature maps. By pooling at different scales, SPPF allows the model to capture features at various levels of abstraction. This can be particularly useful in object detection, where objects of different sizes may need to be detected.  It is an optimized version with the same mathematical functionality of SPP (Spatial Pyramid Pooling) originally used in YOLOv3, but fewer floating-point operations (FLOPs). The specific archetecture used in this project is yolov8n for a balance in speed and accuracy.  

#Optimizations
Possible optimizations are as follows:
- Prunning: 
- Quantization
- Model Distilation
- Layer Fution
