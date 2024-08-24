import torch
from ultralytics import YOLO
from torch.quantization import get_default_qconfig, prepare, conver

# Load your YOLOv8 model
model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")

# Specify the layers you want to quantize
model_quantized = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

# Save the quantized model
torch.save(model_quantized.state_dict(), '"runs/detect/train4_640_full_img/weights/quantized.pt')

model = YOLO("runs/detect/train4_640_full_img/weights/quantized.pt")
validation_results = model.val(data="D:/edith_test/Final.v2i.yolov8/data.yaml", imgsz=640, batch=8, conf=0.5, iou=0.2, device="0")