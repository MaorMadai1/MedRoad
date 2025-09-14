# MedRoad

Cross-Domain Fracture Detection Using Road Damage Models


## About

MedRoad is a project that investigates transfer learning from road-damage detection to bone fracture detection.  
It uses a YOLOv8 model pre-trained on road damage datasets, and integrates LoRA-based convolution layers for parameter-efficient fine-tuning in a new medical imaging domain.

---

## Repository Structure

Here's how the repository is organized:
```
MedRoad/
├── env/ environment.yaml
├── ultralytics/
│ └── engine/
│ └── model.py 
├── YOLOv8_Small_RDD.pt
├── Results/ 
├── TransferLearning/ 
│ └── LoRAConv.py 
│ └── MedRoad.py
│ └── analyze.py
│ └── config.py
│ └── main.py
└── README.md
```

- **env/environment.yaml**: Contains the `*.yaml` file needed to create the conda environment with all required dependencies.  
- **ultralytics/engine/model.py**: This is a patch of the original Ultralytics model file. It must be placed into your local ultralytics installation inside the conda env, replacing or modifying the original.  
- **Results/**: Where our experiment outputs are stored. 
- **TransferLearning/LoRAConv.py**: Implementation of LoRA convolutional layer based on loralib
- **TransferLearning/MedRoad.py**: MedRoad implementation
- **TransferLearning/analyze.py**: Script for figure plotting and results analysis
- **TransferLearning/main.py**: The runfile of the project, contains the paths and hyperparameters
- **TransferLearning/config.py**: Pipe for parameter passing
- **YOLOv8_Small_RDD.pt**: The pre-trained weights on Road Damage Dataset, used as initialization.

---

## Prerequisites

Before running the code, make sure you have:

- **Anaconda / Miniconda** installed  
- GPU with CUDA support (if training or fine-tuning)  

---

## Setup Instructions

Follow these steps to get the code running.

### 1. Create the Conda Environment

1. Navigate to the repository root in your terminal.

2. Create the conda environment from the YAML file in `env/`.

   ```bash
   conda env create -f env/environment.yml

### 2. Update the paths in `TransferLearning/main.py` to match your local setup:

```python
data_path  = "../datasets/GRAZPEDWRI-DX/data.yaml"   # Path to your dataset YAML
weight_path = "../YOLOv8_Small_RDD.pt"              # Path to the pretrained YOLOv8 weights
out_path    = "../BaselineResults"                  # Directory for saving results
```
### 4. Run the Training Script

Activate the conda environment you created earlier, then launch the training:

```bash
conda activate <your_env_name>
python TransferLearning/main.py
```
### 5. (Optional) Analyze Results

You can optionally visualize and summarize your training outputs.

1. **Open the analysis script**  
   Edit `TransferLearning/analyze.py`.

2. **Set the results path**  
   Update the internal variable (or argument, if present) to point to the same directory you configured as `out_path` in `main.py`.

3. **Run the analysis**  
   Execute the script from the project root:
   ```bash
   python TransferLearning/analyze.py
   ```
   

## Credits

We would like to thank the following resources and contributors:

- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics  
  Used as the base object detection framework.
- **LoRA (Low-Rank Adaptation)**: Hu et al., 2021, https://github.com/microsoft/LoRA  
  For the convolutional LoRA layer implementation
- **Road Damage Dataset (RDD2022)**: Arya et al., 2022, https://github.com/oracl4/RoadDamageDetection
  Used for pre-training.
- **GRAZPEDWRI-DX Bone Fracture Dataset**: Nagy et al., 2022, https://www.kaggle.com/datasets/doantrungkien/grazpedwri-dx-dataset
  Used as the target dataset.

