import os
import cv2
import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader, Dataset
from torch.quantization import get_default_qconfig, prepare, convert

class LoadImages(Dataset):
    def __init__(self, images_dir, img_size=(640, 640)):
        self.images_dir = images_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = torch.Tensor(image).reshape(1, 3, 640, 640)
        return image

# Load your YOLOv8 model
model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")

# Switch the model to evaluation mode
model.model.eval()
# Define the quantization configuration
qconfig = get_default_qconfig('fbgemm')
model.model.qconfig = qconfig
# Prepare the model for quantization
model.model = prepare(model.model, inplace=True)

data_loader = LoadImages("D:/edith_test/Final.v2i.yolov8/test/images")

# Calibration step: run inference on a few samples to gather statistics
for images in data_loader:
    #images = torch.tensor(images).permute(0, 3, 1, 2)
    model.model(images)

# Convert the model to its quantized form
quantized_model = convert(model.model, inplace=True)
# # Save the quantized model
#quantized_model.to('cpu')
torch.save(quantized_model.state_dict(), "runs/detect/train4_640_full_img/weights/quantized.pth")

#if __name__ == '__main__':
model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")
model.model.qconfig = get_default_qconfig('fbgemm')
model.model = prepare(model.model, inplace=True)
model.model = convert(model.model, inplace=True)
model.model.load_state_dict(torch.load("runs/detect/train4_640_full_img/weights/quantized.pth"))
validation_results = model.val(data="D:/edith_test/Final.v2i.yolov8/data.yaml", imgsz=640, batch=8, conf=0.5, iou=0.2, device="0")