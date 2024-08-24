from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    # Train the model
    results = model.train(data="D:/edith_test/Final.v2i.yolov8/data.yaml", epochs=100, imgsz=640, batch=8)#, amp=False)