#imports
from ultralytics import YOLO
import loralib as lora
from loralib import Conv2d as LoRAConv2d
from loralib import Linear as LoRALinear

import torch.nn as nn

# from ... import functions


def freeze_all_submodules(layer):
    for module in layer.modules():  # or .children() for just one level
        for param in module.parameters():
            param.requires_grad = False

def Change_model(model):
    #freeze the relevant layers:
    backbone = model.model[:9]
    neck = model.model[10:21]
    head = model.model[22]
    for layer in backbone:
        freeze_all_submodules(layer)

    for layer in backbone:
        freeze_all_submodules(layer)
    # make_lora(neck)
    replace_conv_layer(head, r=2)
    # make_lora(head)
    # replace_conv_layer(neck, r=2)


def replace_conv_layer(model, r=2):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            model.conv = LoRAConv2d(module.in_channels, module.out_channels, kernel_size=module.kernel_size[0], r=r, lora_alpha=0.5, lora_dropout=0.0, merge_weights=False)
        else:
            replace_conv_layer(module, r)


if __name__ == '__main__':
    newdata_path = "..."  # TODO
    model = YOLO("./../RoadDamageDetection/YOLOv8_Small_RDD.pt")
    _data = "C:/Users/User/Documents/Spr 25/Deep/Project/BoneFractureYolo8/data.yaml"

    Change_model(model.model)

    # train
    model.train(data=_data, epochs=10, imgsz=640, batch=16, workers=4, project="BoneFractureYolo8", name="retrain_v1", exist_ok=True, freeze=10)

    print(1)

