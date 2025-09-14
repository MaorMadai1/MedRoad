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
    """
    Extract relevant evaluation metrics from a YOLO model validation result.

    Args:
        results (ultralytics.engine.results.Results): YOLOv8 validation results object.
        run_idx (int): Index of the current run/configuration (used for tracking).

    Returns:
        dict: Dictionary containing the following keys:
            - Run: Run number (1-based)
            - Precision: List of per-class precision values (rounded)
            - Recall: List of per-class recall values (rounded)
            - mAP@0.5: Mean Average Precision at IoU=0.5
            - mAP@0.75: Mean Average Precision at IoU=0.75
            - mAP@0.5:0.95: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
            - Classes: List of class names
            - Preprocess(ms): Time in milliseconds spent on preprocessing
            - Inference(ms): Time in milliseconds spent on inference
            - Postprocess(ms): Time in milliseconds spent on postprocessing
            - Fitness: Optional fitness score from YOLO
    """
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


def evaluate(results_dir, dataset_yaml):
    """
       Evaluate multiple YOLO models stored in a structured directory.

       Iterates over all TRAIN-* folders in the results directory, loads the
       corresponding YOLO model, evaluates it on the provided dataset YAML, and
       collects metrics into a Pandas DataFrame.

       Args:
           results_dir (str): Path to the parent results directory containing TRAIN-* folders.
           dataset_yaml (str): Path to the dataset YAML file for evaluation.

       Returns:
           tuple:
               - df (pd.DataFrame): DataFrame containing evaluation metrics for all runs.
               - configs (list): List of string descriptions of each configuration.
       """
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

        # parser file name:
        parts = folder_name.split("-")
        epochs = next(p.split("=")[1] for p in parts if p.startswith("epochs="))
        last_part = parts[-1].split(",")
        lora = next(p.split("=")[1] for p in last_part if p.startswith("lora="))
        freeze = next(p.split("=")[1] for p in last_part if p.startswith("freeze="))
        configs.append(f"epochs: {epochs}, lora: {lora}, freeze: {freeze}")
        train_path = os.path.join(results_dir, folder_name, "weights", "last.pt")
        if not os.path.exists(train_path):
            print(f"Skipping {folder_name}, last.pt not found")
            continue

        # Load model
        model = YOLO(train_path)
        print(f"Evaluating {folder_name}...")

        # Evaluate on dataset
        results = model.val(data=dataset_yaml, device=device, imgsz=imgsz, batch=bs, save=False, plots=False)

        # Collect metrics
        metrics = get_results(results, i)
        all_metrics.append(metrics)
        i += 1

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(by="mAP@0.5", ascending=False)
    # Save metrics to CSV
    df.to_csv("comparison_metrics.csv", index=False)

    return df, configs


def full_results_figure(df, configs):
    """
    Generate a bar chart figure showing Precision and Recall per class for each configuration.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics from `evaluate()`.
        configs (list): List of string descriptions of each configuration.

    Saves:
        precision_recall_per_config.png: Figure saved in the current working directory.
    """
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
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title(configs[run-1], fontsize=7)

    # Remove unused axes if any
    for i in range(num_runs, len(axes)):
        fig.delaxes(axes[i])

    # Add a single legend for all subplots
    fig.legend(["Precision", "Recall"], loc="upper right")
    fig.suptitle("Precision and Recall per Class for Each Configuration", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    plt.savefig("precision_recall_per_config.png", dpi=300)
    plt.show()


def top_class_figure(df, configs):

    """
    Generate bar charts showing Precision and Recall for selected four top classes across all configurations.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics from `evaluate()`.
        configs (list): List of string descriptions of each configuration.

    Saves:
        precision_recall_per_config_top_classes.png
    """

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
                runs.append(f"{configs[row['Run'] - 1]}")

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
        ax.minorticks_on()
        ax.grid(axis="y", linestyle=":", alpha=0.3, which="minor")

    fig.legend(["Precision", "Recall"])

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("precision_recall_per_config_top_classes.png", dpi=300)
    plt.show()


def top_class_best_configs_figure(df, configs):

    """
    Generate bar charts for the top classes across a set of pre-selected best configurations.

    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics from `evaluate()`.
        configs (list): List of string descriptions of each configuration.

    Saves:
        precision_recall_per_config_top_classes_best_configs.png
    """

    target_classes = ["text", "fracture", "metal"]
    best_configs_var = [[100, 0, None], [32, None, 10], [32, None, 9], [32, 32, 9], [32, 16, 10], [1, 0, 9]]
    best_configs = [f"epochs: {epochs}, lora: {lora}, freeze: {freeze}" for epochs, lora, freeze in best_configs_var]

    # Create figure with one subplot per class
    fig, axes = plt.subplots(1, len(target_classes), figsize=(16, 4), sharey=True)

    for i, cls in enumerate(target_classes):
        ax = axes[i]

        precisions = []
        recalls = []
        runs = []

        for _, row in df.iterrows():
            if cls in row["Classes"] and f"{configs[row['Run'] - 1]}" in best_configs:
                idx = row["Classes"].index(cls)
                precisions.append(row["Precision"][idx])
                recalls.append(row["Recall"][idx])
                runs.append(f"{configs[row['Run'] - 1]}")

        x = np.arange(len(runs))
        width = 0.35

        bars1 = ax.bar(x - width / 2, precisions, width, label="Precision", color="skyblue")
        bars2 = ax.bar(x + width / 2, recalls, width, label="Recall", color="salmon")

        # Add labels above bars
        ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=6)
        ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=6)

        ax.set_title(cls.capitalize(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.minorticks_on()
        ax.grid(axis="y", linestyle=":", alpha=0.3, which="minor")

    fig.legend(["Precision", "Recall"])

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.suptitle("Best Configs, Top Classes: Precision and Recall per Class for Each Configuration", fontsize=16)
    plt.savefig("precision_recall_per_config_top_classes_best_configs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Paths
    results_dir = "../FinalResults"
    dataset_yaml = "../datasets/GRAZPEDWRI-DX/data.yaml"  # replace with your test dataset YAML

    df, configs = evaluate(results_dir, dataset_yaml)

    full_results_figure(df, configs)

    top_class_figure(df, configs)

    top_class_best_configs_figure(df, configs)


