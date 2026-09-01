"""
train_classifier.py - Call-type multi-class classification script (Trumpet vs Roar vs Rumble).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from preprocess import AudioPreprocessor, ElephantDataset


class ElephantCallClassifierCNN(nn.Module):
    """Convolutional Neural Network for multi-class elephant call-type classification."""
    def __init__(self, in_channels=1, num_classes=3):
        super(ElephantCallClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_classifier(dataset_dir, epochs=10, batch_size=8, lr=1e-3, save_path="best_classifier.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocessor = AudioPreprocessor()
    dataset = ElephantDataset(data_dir=dataset_dir, mode="classifier", preprocessor=preprocessor)

    if len(dataset) == 0:
        print("No elephant call-type audio samples found!")
        print("Please place audio files in dataset/elephant/trumpet, dataset/elephant/roar, and dataset/elephant/rumble")
        return

    # Train / Val Split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ElephantCallClassifierCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0.0

    print("Starting Elephant Call-Type Classifier Training...")
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

    print(f"Classifier training complete. Best model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Elephant Call-Type Multi-Class Classifier")
    parser.add_argument("--data_dir", type=str, default="../dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_classifier(dataset_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
