# imports
import torch.nn as nn
import torch.utils.data
from ultralytics import YOLO
import itertools
from LoRAConv import Conv2dNew
import os
import pandas as pd


class MedRoad:
    """
    MedRoad: Cross-domain transfer learning framework for adapting YOLOv8 models
    from road damage detection to bone fracture detection with optional LoRA fine-tuning.

    Attributes:
        model (YOLO): YOLOv8 model loaded from pretrained weights.
        backbone (nn.Module): First layers of YOLOv8 used as feature extractor.
        neck (nn.Module): Middle layers of YOLOv8 (feature aggregation).
        head (nn.Module): Final layers of YOLOv8 (prediction layers).
        lora_r_param (int): Rank of LoRA decomposition applied to Conv layers.

    Methods:
        init_model(): Freeze backbone and optionally replace convs with LoRA.
        extract_layers_from_YOLO(): Extract backbone, neck, head submodules.
        freeze_layer(layer): Freeze parameters in a given module.
        replace_conv_to_LoRa(model, r): Replace Conv2d layers with Conv2dNew.
        get_params_list(layers): Return trainable parameters from given layers.
        get_lora_layers(): Return trainable LoRA params from neck and head.
        train_model(): Train YOLO model with given data and hyperparameters.
        eval_model(): Evaluate YOLO model on validation/test split.
    """
    def __init__(self, weight_path, lora_r_param):
        """
        Initialize MedRoad with a YOLOv8 model.

        Args:
            weight_path (str): Path to YOLOv8 pretrained weights (.pt file).
            lora_r_param (int): LoRA rank parameter (None = no LoRA).
        """
        self.model = YOLO(weight_path)
        self.neck = None
        self.head = None
        self.backbone = None
        self.lora_r_param = lora_r_param
        self.extract_layers_from_YOLO()
        self.init_model()

    def init_model(self):
        """
        Freeze backbone layers and apply LoRA to neck and head if requested.
        """
        MedRoad.freeze_layer(self.backbone)
        if(self.lora_r_param is not None):
            MedRoad.replace_conv_to_LoRa(self.neck, self.lora_r_param)
            MedRoad.replace_conv_to_LoRa(self.head, self.lora_r_param)

    def extract_layers_from_YOLO(self):
        """
        Extract backbone, neck, and head from YOLOv8 architecture.
        """
        self.backbone = self.model.model.model[:9]
        self.neck = self.model.model.model[10:21]
        self.head = self.model.model.model[22]

    @staticmethod
    def freeze_layer(layer):
        """
          Freeze all parameters in the given layer.

          Args:
              layer (nn.Module): PyTorch module to freeze.
          """
        for module in layer.modules():  # or .children() for just one level
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def replace_conv_to_LoRa(model, r=2):
        """
        Recursively replace Conv2d layers with Conv2dNew.

        Args:
            model (nn.Module): Model (neck/head submodule).
            r (int): LoRA rank parameter.
        """
        for name, child in model.named_children():
            if isinstance(child, nn.Conv2d):

                lora_conv = Conv2dNew(child.in_channels, child.out_channels, kernel_size=child.kernel_size[0], r=r,
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
        """
        Return list of trainable parameters from given layers.

        Args:
            layers (list[nn.Module]): Layers to inspect.

        Returns:
            list: Trainable parameters.
        """
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
        """
        Get trainable LoRA parameters from neck and head.

        Returns:
            list: LoRA parameters.
        """
        return MedRoad.get_params_list([self.neck, self.head])

    def train_model(self, data_path, out_path, epochs, imgsz, batch_size, device, **kwargs):
        """
        Train YOLOv8 model with given dataset and parameters.
        """
        return self.model.train(data=data_path,
                                project=out_path,
                                epochs=epochs,
                                imgsz=imgsz,
                                batch=batch_size,
                                device=device,
                                **kwargs)

    def eval_model(self, data_path, imgsz, batch_size, device, split, **kwargs):
        """
        Evaluate YOLOv8 model on a dataset split.

        Args:
            split (str): 'val' or 'test'.
        """
        return self.model.val(data=data_path,
                              imgsz=imgsz,
                              batch=batch_size,
                              device=device,
                              split=split,
                              **kwargs)


def get_results(results, i):
    """
    Extract evaluation metrics from YOLO results object.

    Args:
        results: YOLO validation results object.
        i (int): Run index.

    Returns:
        dict: Collected metrics
    """
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
    """
    Append run information to training_log.txt file.

    Args:
        project_path (str): Output directory.
        dir_name (str): Run directory name.
        i (int): Run index.
        params (dict): Parameters of the run.
    """
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
    """
    Run grid search over hyperparameter combinations.

    Args:
        data_path (str): Path to dataset YAML.
        out_path (str): Output directory.
        weight_path (str): Path to pretrained weights.
        imgsz (int): Image size.
        device (str or torch.device): Device to train on.
        save_period (int): Save frequency.
        param_grid (dict): Dict of lists of parameters to sweep.

    Returns:
        pd.DataFrame: Collected results from all runs.
    """
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