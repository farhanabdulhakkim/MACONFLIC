"""
evaluate.py - Standalone evaluation script for the Elephant Binary Detector.
Evaluates best_detector.pt on the held-out test split from manifest.csv.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocess import AudioPreprocessor, ElephantDataset
from train_detector import ElephantDetectorCNN


def plot_confusion_matrix(cm, class_names, output_path="results/confusion_matrix.png"):
    """Plots and saves a styled confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix (Test Set)')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def evaluate(model_path="best_detector.pt", manifest_path="dataset/manifest.csv", batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}.")

    preprocessor = AudioPreprocessor()
    test_ds = ElephantDataset(manifest_path=manifest_path, split="test", mode="detector", preprocessor=preprocessor)

    if len(test_ds) == 0:
        print("No test samples found in manifest.csv!")
        return

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load model weights
    model = ElephantDetectorCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    print(f"Running evaluation on held-out test set ({len(test_ds)} samples)...")
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            outputs = model(specs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 45)
    print("         TEST SET EVALUATION RESULTS         ")
    print("=" * 45)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("-" * 45)
    print("Classification Report:\n")
    print(classification_report(all_targets, all_preds, target_names=["Non-Elephant", "Elephant"], digits=4))
    print("-" * 45)
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 45)

    plot_confusion_matrix(cm, class_names=["Non-Elephant", "Elephant"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Elephant Binary Detector")
    parser.add_argument("--model_path", type=str, default="best_detector.pt", help="Path to trained model")
    parser.add_argument("--manifest_path", type=str, default="dataset/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    evaluate(model_path=args.model_path, manifest_path=args.manifest_path, batch_size=args.batch_size)