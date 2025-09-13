"""
This file is used for compare between different YOLO models on the same dataset.
We assume the following directory structure:
YOLO_results/
│
├── TRAIN-configA/
│   ├── weights/
│   │   └── last.pt
│   └── ...
│
├── TRAIN-configB/
│   ├── weights/
│   │   └── last.pt
│   └── ...
│
├── TEST-configA/
│   ├── results.json
│   ├── confusion_matrix.png
│   └── ...
│
├── TEST-configB/
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
import torch
import numpy as np

def get_results(results, run_idx):
    return {
        "Run": run_idx + 1,
        "Precision": np.round(results.box.p, 4),
        "Recall": np.round(results.box.r, 4),
        "mAP@0.5": round(results.box.map50, 4),
        "mAP@0.75": round(results.box.map75, 4),
        "mAP@0.5:0.95": round(results.box.map, 4),
        "Classes": [results.names[i] for i in range(len(results.names))],

        # Speed (ms per image)
        "Preprocess(ms)": round(results.speed.get("preprocess", 0), 2),
        "Inference(ms)": round(results.speed.get("inference", 0), 2),
        "Postprocess(ms)": round(results.speed.get("postprocess", 0), 2),

        # Optional: keep fitness if you need it
        "Fitness": results.box.fitness,
    }

if __name__ == "__main__":
    # Paths
    results_dir = "../FinalResults"
    dataset_yaml = "../datasets/GRAZPEDWRI-DX/data.yaml"  # replace with your test dataset YAML

    # Initialize results storage
    all_metrics = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgsz = 640
    bs = 16
    i = 0
    configs = []
    # Iterate over TRAIN folders
    for folder_name in os.listdir(results_dir):
        if not folder_name.startswith("TRAIN-"):
            continue

        configs.append(folder_name.split("batchsize=")[-1].split(",", 1)[-1])
        train_path = os.path.join(results_dir, folder_name, "weights", "last.pt")
        if not os.path.exists(train_path):
            print(f"Skipping {folder_name}, last.pt not found")
            continue

        # Load model
        model = YOLO(train_path)
        print(f"Evaluating {folder_name}...")

        # Evaluate on dataset
        results = model.val(data=dataset_yaml,device=device, imgsz=imgsz, batch=bs,save=False, plots=False)

        # Collect metrics
        metrics = get_results(results, i)
        all_metrics.append(metrics)
        i += 1

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(by="mAP@0.5", ascending=False)
    # Save metrics to CSV
    df.to_csv("comparison_metrics.csv", index=False)


    configs_permuted = [configs[i-1] for i in list(df['Run'])]

    num_runs = df["Run"].nunique()
    ncols = 3
    nrows = int(np.ceil(num_runs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12))
    axes = axes.flatten()

    for idx, run in enumerate(sorted(df["Run"].unique())):
        ax = axes[idx]
        row = df[df["Run"] == run].iloc[0]  # get the row for this run
        classes = row["Classes"]
        precision = row["Precision"]
        recall = row["Recall"]
        x = np.arange(len(classes))
        width = 0.35

        ax.bar(x - width / 2, precision, width, label="Precision", color='skyblue')
        ax.bar(x + width / 2, recall, width, label="Recall", color='salmon')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(configs[run-1])

    # Remove unused axes if any
    for i in range(num_runs, len(axes)):
        fig.delaxes(axes[i])

    # Add a single legend for all subplots
    fig.legend(["Precision", "Recall"], loc="upper right")
    fig.suptitle("Precision and Recall per Class for Each Configuration", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    plt.savefig("precision_recall_per_config.png", dpi=300)
    plt.show()

    target_classes = ["text", "fracture", "metal", "periostealreaction"]

    # Create figure with one subplot per class
    fig, axes = plt.subplots(1, len(target_classes), figsize=(16, 4), sharey=True)

    for i, cls in enumerate(target_classes):
        ax = axes[i]

        precisions = []
        recalls = []
        runs = []

        for _, row in df.iterrows():
            if cls in row["Classes"]:
                idx = row["Classes"].index(cls)
                precisions.append(row["Precision"][idx])
                recalls.append(row["Recall"][idx])
                runs.append(f"{configs[row['Run']-1]}")

        x = np.arange(len(runs))
        width = 0.35

        ax.bar(x - width / 2, precisions, width, label="Precision", color="skyblue")
        ax.bar(x + width / 2, recalls, width, label="Recall", color="salmon")

        ax.set_title(cls.capitalize(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    fig.legend(["Precision", "Recall"])

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

    # Create a figure with 2 bar charts (avg Precision and avg Recall)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Precision
    axes[0].bar(configs_permuted, df["Precision"], color="skyblue")
    axes[0].set_title("Precision")
    axes[0].set_xticklabels(configs_permuted, rotation=45, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.7)

    # Recall
    axes[1].bar(configs_permuted, df["Recall"], color="lightgreen")
    axes[1].set_title("Recall")
    axes[1].set_xticklabels(configs_permuted, rotation=45, ha="right")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.title("Avg Precision and Avg Recall", fontsize=14)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig("precision_recall_comparison.png")

