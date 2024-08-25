from ultralytics import YOLO

# # Load the YOLOv8 model
# model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")

# # Export the model to TFLite format
# model.export(format="tflite")  # creates 'yolov8n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("runs/detect/train4_640_full_img/weights/best_saved_model/saved_model.pb")

if __name__ == '__main__':
    # validation
    validation_results = tflite_model.val(data="D:/edith_test/Final.v2i.yolov8/data.yaml", imgsz=640, batch=8, conf=0.5, iou=0.2)