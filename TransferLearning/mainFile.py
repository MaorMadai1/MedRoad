#imports
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from ultralytics import YOLO
from loralib import Conv2d as LoRAConv2d
from pathlib import Path
import time


class LoRATraining:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.neck = None
        self.head = None
        self.backbone = None
        self.init_model()

    def init_model(self):
        self.extract_layers_from_YOLO()
        LoRATraining.freeze_layer(self.backbone)

        # for layer in neck:
        LoRATraining.replace_conv_to_LoRa(self.neck)
        # for layer in head:
        LoRATraining.replace_conv_to_LoRa(self.head)

    def extract_layers_from_YOLO(self):
        self.backbone = self.model.model.model[:9]
        self.neck = self.model.model.model[10:21]
        self.head = self.model.model.model[22]

    def train_model(self, data_path, epochs, imgsz, batch_size, device, **kwargs):
        dir_name = f"retrain_v1_epochs_{epochs}_imgsz_{imgsz}_batchsize_{batch_size}"
        start_time = time.time()
        metrics = self.model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch_size, device=device,
                                   project="BoneFractureYolo8", name=dir_name, exist_ok=True, **kwargs)
        best_model = YOLO(Path(self.model.trainer.best))
        end_time = time.time()
        dt = end_time - start_time
        print(f"Training complete in {dt // 60:.0f} m {dt % 60:.0f} s")
        return best_model, metrics

        # TODO: @maor @maayan - do we need other params? -lr, -momentum, -data-aug, -optimizer, -criterion,

    def eval_model(self, data_path, imgsz, batch_size, device, split, **kwargs):
        start_time = time.time()
        metrics = self.model.val(data=data_path, imgsz=imgsz, batch=batch_size, device=device, split=split, **kwargs)
        end_time = time.time()
        dt = end_time - start_time
        print(f"Evaluation complete in {dt // 60:.0f} m {dt % 60:.0f} s")
        return metrics

    @staticmethod
    def freeze_layer(layer):
        for module in layer.modules():  # or .children() for just one level
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def replace_conv_to_LoRa(model, r=2):
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                model.conv = LoRAConv2d(module.in_channels, module.out_channels, kernel_size=module.kernel_size[0], r=r,
                                        lora_alpha=0.5, lora_dropout=0.0, merge_weights=False)
            else:
                LoRATraining.replace_conv_to_LoRa(module, r)

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
        return LoRATraining.get_params_list([self.neck, self.head])

    @staticmethod
    def DataPreProccess(data_dir, batch_size, seed=0):

        assert (batch_size >= 64)  # batchNorm works well for batch_size>=64
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.1, 0.1, 0.1], [0.1, 0.1, 0.1])  # TODO: @maor - find the corerct mean & std
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.1, 0.1, 0.1], [0.1, 0.1, 0.1])  # TODO: @maor - add transforms?
            ])
        }

        # TODO - @maayan
        # trainset = load_dataset()
        # valset = load_dataset()
        train_dataloader = None  # torch.utils.data.DataLoader(trainset, batch_size,shuffle=True)
        val_dataloader = None  # torch.utils.data.DataLoader(valset,batch_size,shuffle=True)
        return train_dataloader, val_dataloader


if __name__ == '__main__':
    # params:
    datapath = "./../datasets/BoneFractureYolo8/data.yaml"
    lr = 0.00001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss() #TODO - isn't it sometihng else?
    epochs = 20
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data-preprocess:
    # dataloaders = DataPreProccess(datapath, batch_size, 1234)
    imgsize = 640 #TODO: @maor find the size from the dataset

    # Init-the model
    weight_path = "./../RoadDamageDetection/YOLOv8_Small_RDD.pt"
    lora_trainer = LoRATraining(weight_path=weight_path)
    lora_trainer.get_lora_layers()
    # optimizer = torch.optim.SGD(paramsToUpdate, lr, momentum)
    best_model, hist = lora_trainer.train_model(data_path=datapath, epochs=epochs, imgsz=imgsize, batch_size=batch_size,
                                                device=device)

    # Ideas:
    #1) "clean" train - just of this dataset
    #2) freeze all layers - what to do with the output dim? (num_of_classes)
    #3) use DoRA instead of LoRA

    # newdata_path = "..."  # TODO
    # # train
    # model.train(data=_data, epochs=10, imgsz=640, batch=16, workers=4, project="BoneFractureYolo8", name="retrain_v1", exist_ok=True, freeze=10)

























