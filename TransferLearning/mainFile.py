#imports
from ultralytics import YOLO
# from ... import functions
newdata_path = "..." #TODO

prev_model = YOLO("RoadDamageDetection/YOLOv8_Small_RDD.pt")
model = PrepareModel(prev_model)
model.train()
model.val()

def PrepareModel():
    freezeLayers()
    LoraLayers()
    MoreSpicyStuff()
