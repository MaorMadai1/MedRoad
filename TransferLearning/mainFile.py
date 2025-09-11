# imports
import torch.nn as nn
import torch.utils.data
from ultralytics import YOLO
import itertools
from loralib import Conv2d as LoRAConv2d
import loralib as lora
import os
import pandas as pd
import time
import config


class MedRoad:
    def __init__(self, weight_path, lora_r_param):
        self.model = YOLO(weight_path)
        self.neck = None
        self.head = None
        self.backbone = None
        self.lora_r_param = lora_r_param
        self.extract_layers_from_YOLO()
        self.init_model()

    def init_model(self):
        MedRoad.freeze_layer(self.backbone)
        if(self.lora_r_param is not None):
            MedRoad.replace_conv_to_LoRa(self.neck, self.lora_r_param)
            MedRoad.replace_conv_to_LoRa(self.head, self.lora_r_param)

    def extract_layers_from_YOLO(self):
        self.backbone = self.model.model.model[:9]
        self.neck = self.model.model.model[10:21]
        self.head = self.model.model.model[22]

    @staticmethod
    def freeze_layer(layer):
        for module in layer.modules():  # or .children() for just one level
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def replace_conv_to_LoRa(model, r=2):
        for name, child in model.named_children():
            if isinstance(child, nn.Conv2d):

                lora_conv = LoRAConv2d(child.in_channels, child.out_channels, kernel_size=child.kernel_size[0], r=r,
                                        lora_alpha=0.5, lora_dropout=0.0, stride=child.stride,
                                        padding = child.padding, dilation = child.dilation, bias = child.bias is not None, groups = child.groups,
                                       merge_weights=False)
                lora_conv.conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    lora_conv.conv.bias.data.copy_(child.bias.data)
                setattr(model, name, lora_conv)
            else:
                MedRoad.replace_conv_to_LoRa(child, r)


    @staticmethod
    def get_params_list(layers):
        print("---Params to update---")
        params = []
        for layer in layers:
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    params.append(param)
                    print("\t", name)
        print("-------Done-------")
        return params

    def get_lora_layers(self):
        return MedRoad.get_params_list([self.neck, self.head])

    def train_model(self, data_path, out_path, epochs, imgsz, batch_size, device, **kwargs):
        return self.model.train(data=data_path,
                                project=out_path,
                                epochs=epochs,
                                imgsz=imgsz,
                                batch=batch_size,
                                device=device,
                                **kwargs)

    def eval_model(self, data_path, imgsz, batch_size, device, split, **kwargs):
        return self.model.val(data=data_path,
                              imgsz=imgsz,
                              batch=batch_size,
                              device=device,
                              split=split,
                              **kwargs)


def get_results(results, i):
    output = {
        # Core metrics
        "Run": i + 1,
        "AP50": results.box.ap50,
        "AP": results.box.ap,
        "mean-precision": results.box.mp,
        "mean-recall": results.box.mr,
        "mAP@0.5": round(results.box.map50, 4),
        "mAP@0.75": round(results.box.map75, 4),
        "mAP@0.5:0.95": round(results.box.map, 4),

        # Speed
        "Preprocess(ms)": round(results.speed.get('preprocess', 0), 2),
        "Inference(ms)": round(results.speed.get('inference', 0), 2),
        "Postprocess(ms)": round(results.speed.get('postprocess', 0), 2),

        # Optional: fitness
        "Fitness": results.box.fitness,
    }

    return output


def update_log(project_path, dir_name, i, params):
    # Path to log file
    log_file = os.path.join(project_path, "training_log.txt")

    # Build a log string
    log_string = dir_name
    log_string += "Params:\n" + "\n".join([f"  {k}: {v}" for k, v in params.items()])
    log_string += "\n---\n"

    # Append to log file
    with open(log_file, "a") as f:
        f.write(log_string)


def grid_search(data_path, out_path, weight_path, imgsz, device, save_period, param_grid):
    # Ensure output dir exists & right params:
    required_keys = {"epochs", "batch_size", "lora_r_param"}
    missing_keys = required_keys - param_grid.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    # Create parameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total combinations: {len(param_combinations)}")

    # Store results
    all_results = []
    for i, params in enumerate(param_combinations):
        print(f"\n🔍 Running combination {i + 1}/{len(param_combinations)}: {params}")
        dir_name = f"Run_{i + 1}_" + "_".join([f"{k}{v}" for k, v in params.items()])
        epochs = params.pop("epochs")
        batch_size = params.pop("batch_size")
        lora_r_param = params.pop("lora_r_param")
        # The actual run:
        model = MedRoad(weight_path, lora_r_param)
        model.train_model(data_path=data_path,
                          out_path=out_path,
                          epochs=epochs,
                          imgsz=imgsz,
                          batch_size=batch_size,
                          device=device,
                          name=dir_name,
                          **params)
        val_res = model.eval_model(data_path=data_path,
                                   out_path=out_path,
                                   imgsz=imgsz,
                                   batch_size=batch_size,
                                   device=device,
                                   split='val')
        # update_stats:
        all_results.append(get_results(val_res, i))
        update_log(out_path, dir_name, i, params)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(out_path, "grid_search_results.csv"), index=False)
    return df

if __name__ == '__main__':
    # params:
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
