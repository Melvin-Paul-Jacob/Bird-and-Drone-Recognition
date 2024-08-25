import cv2
import numpy as np
from ultralytics import YOLO

class EuclideanDistTracker:
    def __init__(self, min_dist=200, max_disp=100):
        # Store the center positions of the objects
        self.min_dist = min_dist
        self.max_disp = max_disp
        self.center_points = {}
        self.disappeared = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
            
    def centroid(self, rect):
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return np.array([cx, cy])

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []
        # Get center point of new object
        for rect in objects_rect:
            area = (rect[1]-rect[3])*(rect[0]-rect[2])
            center = self.centroid(rect)
            # Find out if that object was detected already
            same_object_detected = False
            obs = np.array(list(self.center_points.values()))
            ids = np.array(list(self.center_points.keys()))
            if len(obs)>0:
                dists = np.linalg.norm(center-obs, axis=1)
                closest_indx = np.argmin(dists)
                if dists[closest_indx] < self.min_dist:
                    id = ids[closest_indx]
                    self.center_points[id] = center
                    #print(self.center_points)
                    objects_bbs_ids.append([rect, id,area])
                    same_object_detected = True
                    self.disappeared[id] = 0

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = center
                objects_bbs_ids.append([rect, self.id_count,area])
                self.disappeared[self.id_count] = 0
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            re, object_id, ar = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        disap = [x for x in list(self.center_points.keys()) if x not in list(new_center_points.keys())]
        for i in disap:
            self.disappeared[i] += 1
            if self.disappeared[i]<self.max_disp:
                new_center_points[i] = self.center_points[i]
        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

# Model
model = YOLO("runs/detect/train4_640_full_img/weights/best.pt",verbose=False)
cap = cv2.VideoCapture("test_vid_bird.mp4")
frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
   print("error opening camera")
   exit()

tracker = EuclideanDistTracker()

time_accum = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("error in retrieving frame")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #results = model.track(frame_rgb,imgsz=640,iou=0.2,verbose=False, persist=True)
    results = model(frame_rgb,imgsz=640,iou=0.2,verbose=False)#,conf=0.6)
    boxes = results[0].boxes.xyxy.cpu()
    bboxs = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        bboxs.append([x1,y1,x2,y2])
    objects_bbs_ids = tracker.update(bboxs)
    for bbox in objects_bbs_ids:
        cv2.rectangle(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[0][2]), int(bbox[0][3])), (255,255,255), thickness=2)
        cv2.putText(frame, str(bbox[1]), (int(bbox[0][0]), int(bbox[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    inference_time = float(results[0].speed['inference'])
    time_accum = time_accum + inference_time
    #annotated_frame = results[0].plot()
    #annotated_frame = cv2.resize(annotated_frame, (640,480))
    frame = cv2.resize(frame, (640,480))
    #bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Screen', frame)
    if cv2.waitKey(1) == ord('q'):
        break

print(f"Average Inference time: {time_accum/frame_no:.2f} ms")
cap.release()
cv2.destroyAllWindows()