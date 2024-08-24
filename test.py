from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train4/weights/best.pt")

if __name__ == '__main__':
    # validation
    validation_results = model.val(data="D:/edith_test/Final.v2i.yolov8/data.yaml", imgsz=640, batch=8, conf=0.25, iou=0.5, device="0")