We used the pretrain model, and retrain it for 10 epochs while freezing the Backbone.
This is what we ran:

# loading the RoadDamage weights and the new data:
_data = "C:/Users/User/Documents/Spr 25/Deep/Project/BoneFractureYolo8/data.yaml"
weight_path = "C:/Users/User/Documents/Spr 25/Deep/Project/RoadDamageDetection/models/YOLOv8_Small_RDD.pt"

model = YOLO(weight_path)

# retraining:
model.train(data=_data, epochs=10, imgsz=640, batch=16, workers=4, project="BoneFractureYolo8", name="retrain_v1", exist_ok=True, freeze=10)

# test:

model2 = YOLO(r"C:\Users\User\Documents\Spr 25\Deep\Project\BaselineResults\weights\best.pt")
res = model2.val(data=_data, split='test', save_dir='BoneFractureYolo8/test_results')
print(res.save_dir)
