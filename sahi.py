from sahi.predict import predict

predict(
    model_type="yolov8",
    model_path="runs/detect/train4_640_full_img/weights/best.pt",
    model_device=0,  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="D:/edith_test/Final.v2i.yolov8/valid/images",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)