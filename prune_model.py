import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import numpy as np

def prune_model(model,amount=0.3):
    for module in model.modules():
        if isinstance(module,torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
        # elif isinstance(module, torch.nn.Linear):
        #     prune.l1_unstructured(module, name='weight', amount=0.4)
        #     prune.remove(module, "weight")
    return model

#load model

if __name__ == '__main__':
    #for i in np.arange(0.1, 0.5, 0.1):
    model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")
    torch_model = model.model
    print("pruning model")
    pruned_model = prune_model(torch_model, amount=0.1)
    print("model pruned")
    model.model = pruned_model
    #save pruned model
    model.save("runs/detect/train4_640_full_img/weights/pruned.pt")
    # model = YOLO("runs/detect/train4_640_full_img/weights/pruned.pt")
    # print(str(i),"----------------")
    # validation_results = model.val(data="D:/edith_test/Final.v2i.yolov8/data.yaml", imgsz=640, batch=8, conf=0.5, iou=0.2, device="0")