import cv2
from ultralytics import YOLO

# Model
model = YOLO("runs/detect/train4_640_full_img/weights/pruned.pt",verbose=False)
cap = cv2.VideoCapture("test_vid_bird.mp4")
frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
   print("error opening camera")
   exit()
time_accum = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("error in retrieving frame")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb,imgsz=640,iou=0.2,verbose=False)#,conf=0.6)
    inference_time = float(results[0].speed['inference'])
    time_accum = time_accum + inference_time
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (640,480))
    bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Screen', bgr)
    if cv2.waitKey(1) == ord('q'):
        break

print(f"Average Inference time: {time_accum/frame_no:.2f} ms")
cap.release()
cv2.destroyAllWindows()