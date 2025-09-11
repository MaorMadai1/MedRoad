from mainFile import MedRoad
import torch.utils.data
import loralib as lora
import config
import time


data_path = "../datasets/GRAZPEDWRI-DX/data.yaml"
weight_path = "../RoadDamageDetection/YOLOv8_Small_RDD.pt"
out_path = "../BaselineResults"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgsz = 640

#sweeps:
epochs = [32]
batch_size = [16]
lorap = [32]
freezes = [10, 9]
# Global_R_LoRA = 16
#cos_lr = [True,False]
#freeze = [1,2,3]
#lr0 = [1e-5,1e-5,1e-5]
#lrf = [0.01,0.001]


# Init-the model
for e in epochs:
    for bs in batch_size:
        for r in lorap:
            for f in freezes:
                t = time.strftime("%Y-%m-%d_%H-%M-%S")
                config.Global_R_LoRA = r # set the r for LoRA so it will be update in Ultralytics
                model = MedRoad(weight_path, lora_r_param=r)
                model.model.train(data=data_path,
                                  project=out_path,
                                  name=f"TRAIN-time={t}-epochs={e}-batchsize={bs},lora={r},freeze={f}",
                                  epochs=e,
                                  imgsz=imgsz,
                                  batch=bs,
                                  device=device,
                                  save=False,
                                  freeze=f
                                  )

                print("**********************")
                for name, module in model.model.named_modules():
                    if isinstance(module, lora.Conv2dNew):
                        print("LoRAConv2d found at:", name)
                total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                print(f'Total number of trainable parameters: {total_params}')

                model.model.val(data=data_path,
                                project=out_path,
                                name=f"TEST-time={t}-epochs={e}-batchsize={bs},lora={r},freeze={f}",
                                imgsz=imgsz,
                                device=device,
                                batch=bs,
                                split='test',
                                save_txt=True,
                                save_conf=True,
                                save_json=True,
                                verbose=True,
                                plots=True)
