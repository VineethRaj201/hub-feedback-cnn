"""
eval.py

Evaluates and compares the BaselineCNN and HubFeedbackCNN on clean
and corrupted MNIST test images.

Usage:
    python eval.py

Expects saved weights:
    baseline_cnn.pt
    hub_feedback_cnn.pt
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.baseline_cnn import BaselineCNN
from models.hub_feedback_cnn import HubFeedbackCNN
from corruptions import CORRUPTIONS


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device, corruption_fn):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = corruption_fn(x)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    device = get_device()
    print(f"Using device: {device}\n")

    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    # Load models
    baseline = BaselineCNN().to(device)
    baseline.load_state_dict(torch.load("baseline_cnn.pt", map_location=device))

    hub_model = HubFeedbackCNN(cycles=3, hub_dim=64).to(device)
    hub_model.load_state_dict(torch.load("hub_feedback_cnn.pt", map_location=device))

    models = {
        "BaselineCNN":     baseline,
        "HubFeedbackCNN":  hub_model,
    }

    # Header
    col_w = 18
    header = f"{'Corruption':<{col_w}}" + "".join(f"{name:>{col_w}}" for name in models)
    print(header)
    print("-" * len(header))

    # Evaluate each model on each corruption
    for corruption_name, corruption_fn in CORRUPTIONS.items():
        row = f"{corruption_name:<{col_w}}"
        for model in models.values():
            acc = evaluate(model, test_loader, device, corruption_fn)
            row += f"{acc * 100:>{col_w - 1}.2f}%"
        print(row)


if __name__ == "__main__":
    main()
