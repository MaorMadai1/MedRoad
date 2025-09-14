import torch.utils.data
import time
import config
from MedRoad import MedRoad

if __name__ == '__main__':
    # params:
    data_path = "../datasets/GRAZPEDWRI-DX/data.yaml"
    weight_path = "../YOLOv8_Small_RDD.pt"
    out_path = "../BaselineResults"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgsz = 640

    # sweeps parameters:
    epochs = [10]
    batch_size = [16]
    lorap = [4]
    freezes = [9]

    # other hyperparamaters can be add here

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
                                      freeze=f is not None
                                      )
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
