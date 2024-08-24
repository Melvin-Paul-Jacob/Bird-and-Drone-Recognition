import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model, amount=0.3):
    for module in model.modules():
        if isinstance(module,torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model

#load model

if __name__ == '__main__':
    for i in range(0.1, 0.5, 0.1):
        model = YOLO("runs/detect/train4_640_full_img/weights/best.pt")
        torch_model = model.model
        print("pruning model")
        pruned_model = prune_model(torch_model, amount=0.2)
        print("model pruned")
        model.model = pruned_model
        #save pruned model
        # model.save("runs/detect/train4/weights/pruned.pt")
