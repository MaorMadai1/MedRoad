#imports
import torch.nn as nn
import torch.utils.data
from torchvision import transforms
from ultralytics import YOLO
from loralib import Conv2d as LoRAConv2d
from pathlib import Path
import time


def FreezeLayer(layer):
    for module in layer.modules():  # or .children() for just one level
        for param in module.parameters():
            param.requires_grad = False

def ReplaceConvToLora(model, r=2):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            model.conv = LoRAConv2d(module.in_channels, module.out_channels, kernel_size=module.kernel_size[0], r=r, lora_alpha=0.5, lora_dropout=0.0, merge_weights=False)
        else:
            ReplaceConvToLora(module, r)


def DataPreProccess(data_dir, batch_size,seed = 0):

    assert(batch_size>=64) #batchNorm works well for batch_size>=64
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.1, 0.1, 0.1], [0.1, 0.1, 0.1]) #TODO: @maor - find the corerct mean & std
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.1, 0.1, 0.1], [0.1, 0.1, 0.1]) #TODO: @maor - add transforms?
        ])
    }

    #TODO - @maayan
    #trainset = load_dataset()
    #valset = load_dataset()
    train_dataloader = None #torch.utils.data.DataLoader(trainset, batch_size,shuffle=True)
    val_dataloader = None #torch.utils.data.DataLoader(valset,batch_size,shuffle=True)
    return train_dataloader,val_dataloader


def ExtractLayersFromYOLO(model):
    backbone = model.model.model[:9]
    neck = model.model.model[10:21]
    head = model.model.model[22]

    return backbone, neck, head

def GetParamsList(layers):
    print("---Params to update---")
    params = []
    for layer in layers:
        for name, param in layer.named_parameters():
            if param.requires_grad:
                params.append(param)
                print("\t", name)
    print("-------Done-------")
    return params


def InitModel(model):
    backbone, neck, head = ExtractLayersFromYOLO(model)
    FreezeLayer(backbone)
    # for layer in neck:
    ReplaceConvToLora(neck)
    # for layer in head:
    ReplaceConvToLora(head)

    return model

def TrainModel(model: YOLO, data_path, epochs, imgsz, batch_size, device, **kwargs):
    start_time = time.time()
    metrics = model.train(data=datapath, epochs=epochs, imgsz=imgsz, batch=batch_size, device=device,
                          project="BoneFractureYolo8", name="retrain_v1", exist_ok=True, **kwargs)
    best_model = YOLO(Path(model.trainer.best))
    end_time = time.time()
    dt = end_time - start_time
    print(f"Training complete in {dt//60:.0f} m {dt%60:.0f} s")
    return best_model, metrics

    #TODO: @maor @maayan - do we need other params? -lr, -momentum, -data-aug, -optimizer, -criterion,

def EvalModel():
    return None


if __name__ == '__main__':
    #params:
    datapath = "./../datasets/BoneFractureYolo8/data.yaml"
    lr = 0.00001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss() #TODO - isn't it sometihng else?
    epochs = 20
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Data-preprocess:
    dataloaders = DataPreProccess(datapath, batch_size, 1234)
    imgsize = 640 #TODO: @maor find the size from the dataset

    #Init-the model
    model = YOLO("./../RoadDamageDetection/YOLOv8_Small_RDD.pt")
    InitModel(model)
    bb, neck, head = ExtractLayersFromYOLO(model)
    paramsToUpdate = GetParamsList([neck,head])
    optimizer = torch.optim.SGD(paramsToUpdate, lr, momentum)
    best_model, hist = TrainModel(model, datapath, epochs, imgsize, batch_size, device)


    #Ideas:
    #1) "clean" train - just of this dataset
    #2) freeze all layers - what to do with the output dim? (num_of_classes)
    #3) use DoRA instead of LoRA

    # newdata_path = "..."  # TODO
    # # train
    # model.train(data=_data, epochs=10, imgsz=640, batch=16, workers=4, project="BoneFractureYolo8", name="retrain_v1", exist_ok=True, freeze=10)

























