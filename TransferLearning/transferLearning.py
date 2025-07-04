#TODO: do we need more imports?
from ultralytics import YOLO
#TODO: fix the path!
weight_path = "/home/oracl4/project/RoadDamageDetection/models/YOLOv8_Small_RDD.pt"
data_path = ".." #TODO: add datapath!
model = YOLO(weight_path)

def freeze_layers(model, option: str,N = None):
    # This freezes models layers by option
    total_layers = len(model.model)
    if N != None & N>total_layers:
        raise Exception("N is larger than number of layers")
    if option is "backbone":
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    if option is "layers":
        count = 0
        for layer in model.model.modules():  # model.model is nn.Module
            if any(p.requires_grad for p in layer.parameters(recurse=False)):
                for p in layer.parameters(recurse=False):
                    p.requires_grad = False
                count += 1
                if count >= N:
                    break

def model_train(model,epochs):
    model.train(data='data_path', epochs=epochs)