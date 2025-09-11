"""
This file is used for compare between different YOLO models on the same dataset.
We assume the following directory structure:
YOLO_results/
│
├── TRAIN_configA/
│   ├── weights/
│   │   └── last.pt
│   └── ...
│
├── TRAIN_configB/
│   ├── weights/
│   │   └── last.pt
│   └── ...
│
├── TEST_configA/
│   ├── results.json
│   ├── confusion_matrix.png
│   └── ...
│
├── TEST_configB/
│   ├── results.json
│   ├── confusion_matrix.png
│   └── ...
│
└── ... (more TRAIN_* and TEST_* folders)
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

# Paths
results_dir = "./YOLO_results"
dataset_yaml = "dataset.yaml"  # replace with your test dataset YAML

# Initialize results storage
all_metrics = []

# Iterate over TRAIN folders
for folder_name in os.listdir(results_dir):
    if not folder_name.startswith("TRAIN_"):
        continue

    train_path = os.path.join(results_dir, folder_name, "weights", "last.pt")
    if not os.path.exists(train_path):
        print(f"Skipping {folder_name}, last.pt not found")
        continue

    # Load model
    model = YOLO(train_path)
    print(f"Evaluating {folder_name}...")

    # Evaluate on dataset
    results = model.val(data=dataset_yaml, save=False, plots=False)

    # Collect metrics
    metrics = {
        "config": folder_name,
        "precision": results.metrics.get("precision", None),
        "recall": results.metrics.get("recall", None),
        "mAP_50": results.metrics.get("map_50", None),
        "mAP_50_95": results.metrics.get("map_50_95", None),
        "fitness": results.metrics.get("fitness", None),
        "cls_loss": results.loss.get("cls_loss", None),
        "box_loss": results.loss.get("box_loss", None),
        "dfl_loss": results.loss.get("dfl_loss", None),
    }
    all_metrics.append(metrics)

# Convert to DataFrame
df = pd.DataFrame(all_metrics)
df = df.sort_values(by="mAP_50", ascending=False)
print(df)

# Save metrics to CSV
df.to_csv("comparison_metrics.csv", index=False)

# Plot comparison of key metrics
plt.figure(figsize=(12, 6))
x = df["config"]

plt.plot(x, df["mAP_50"], marker='o', label="mAP@0.5")
plt.plot(x, df["mAP_50_95"], marker='o', label="mAP@0.5:0.95")
plt.plot(x, df["precision"], marker='o', label="Precision")
plt.plot(x, df["recall"], marker='o', label="Recall")

plt.xticks(rotation=45, ha="right")
plt.ylabel("Score")
plt.title("YOLOv8 Configuration Comparison")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("comparison_plot.png")
plt.show()