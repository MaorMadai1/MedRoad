#imports
from ultralytics import YOLO
# from ... import functions
newdata_path = "..." #TODO

model = YOLO("./../RoadDamageDetection/YOLOv8_Small_RDD.pt")


def Change_model(model):
    #freeze the relevant layers:
    backbone = model.model[:9]
    neck = model.model[10:21]
    head = model.model[22]
    for layer in backbone:
        freeze_all_submodules(layer)
    # make_lora(neck)
    # make_lora(head)

def freeze_all_submodules(layer):
    for module in layer.modules():  # or .children() for just one level
        for param in module.parameters():
            param.requires_grad = False
