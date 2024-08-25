import cv2
import numpy as np
from ultralytics import YOLO
from yolov8_gradcam.yolo_cam.eigen_cam import EigenCAM
from yolov8_gradcam.yolo_cam.utils.image import show_cam_on_image, scale_cam_image

# Load a model
model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")
image = cv2.imread("D:/edith_test/Final.v2i.yolov8/valid/images/Birds_in_the_Sky_-TK1-_JPG_jpg.rf.dbbd76e88d2539fd740b8e75eb0f35d6.jpg")
image = cv2.resize(image, (640, 640))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgb_img = image.copy()
image = np.float32(image) / 255
target_layers =[model.model.model[-2]]

cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
#cv2.imshow("img", cam_image)

cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
cv2.imshow("img1", cam_image)
cv2.imwrite("gradcam.png",cam_image)
cv2.waitKey(0)
cv2.destroyAllWindows()