import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.baseline_cnn import BaselineCNN
from models.hub_feedback_cnn import HubFeedbackCNN


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train():
    device = get_device()
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    use_hub = False  # flip to False to train baseline
    if use_hub:
        model = HubFeedbackCNN(cycles=3, hub_dim=64).to(device)
    else:
        model = BaselineCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: train accuracy = {acc:.4f}")

    torch.save(model.state_dict(), "baseline_cnn.pt")
    print("Saved baseline_cnn.pt")


if __name__ == "__main__":
    train()