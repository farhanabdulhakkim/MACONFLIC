"""
train_detector.py - Binary classification script (Elephant vs Non-Elephant).
Now consumes the leak-safe manifest.csv dataset splits.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import AudioPreprocessor, ElephantDataset


class ElephantDetectorCNN(nn.Module):
    """Convolutional Neural Network for binary audio detection (spectrogram input)."""
    def __init__(self, in_channels=1, num_classes=2):
        super(ElephantDetectorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_detector(dataset_dir, epochs=10, batch_size=8, lr=1e-3, save_path="best_detector.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocessor = AudioPreprocessor()
    manifest_path = os.path.join(dataset_dir, "manifest.csv")

    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}! Please run prepare_dataset.py first.")
        return

    # Load explicit, leak-safe splits
    train_ds = ElephantDataset(manifest_path=manifest_path, split="train", mode="detector", preprocessor=preprocessor)
    val_ds = ElephantDataset(manifest_path=manifest_path, split="val", mode="detector", preprocessor=preprocessor)
    
    train_size = len(train_ds)
    val_size = len(val_ds)

    if train_size == 0 or val_size == 0:
        print("Dataset splits are empty! Check your manifest.csv.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ElephantDetectorCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0

    print(f"Starting Binary Detector Training... (Train: {train_size} | Val: {val_size})")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * specs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        train_acc = train_correct / train_size if train_size > 0 else 0

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * specs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / val_size if val_size > 0 else train_acc

        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss/train_size:.4f} Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"Detector training complete. Best model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Elephant Binary Detector")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_detector(dataset_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)