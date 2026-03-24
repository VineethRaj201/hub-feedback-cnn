import torch
import torch.nn as nn
import torch.nn.functional as F

from .hub import Hub


class HubFeedbackCNN(nn.Module):
    """
    Baseline CNN + a hub state that feeds back via feature-map gating.
    Runs K cycles: (conv -> pool -> conv -> pool -> hub update -> gate features) x K,
    then classifies.
    """
    def __init__(self, num_classes=10, hub_dim=64, cycles=3):
        super().__init__()
        self.cycles = cycles
        self.hub_dim = hub_dim

        # Same conv backbone idea as baseline
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # summarize the conv2 feature maps (64 channels) into a vector
        # via global average pooling => (B, 64)
        feat_dim = 64

        self.hub = Hub(feat_dim=feat_dim, hub_dim=hub_dim)

        # Feedback/gating: hub state -> per-channel gate for conv2 features (64 channels)
        self.gate = nn.Linear(hub_dim, 64)

        # Classifier head (same shape as baseline)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_once(self, x, h):
        """
        One cycle:
        - run convs
        - update hub from feature summary
        - gate conv2 feature maps using hub
        Returns: gated features (B,64,7,7), updated hub
        """
        x = self.pool(F.relu(self.conv1(x)))     # (B,32,14,14)
        f = self.pool(F.relu(self.conv2(x)))     # (B,64,7,7)

        # summarize features -> s_t (B,64)
        s = f.mean(dim=(2, 3))

        # update hub state
        h = self.hub(s, h)

        # feedback gate (B,64) -> (B,64,1,1)
        g = torch.sigmoid(self.gate(h)).unsqueeze(-1).unsqueeze(-1)

        # apply gate (feedback)
        f = f * (1.0 + g)   # (1 + g) keeps it mild; not zeroing everything

        return f, h

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        h = self.hub.init_state(batch_size, device)

        # run K cycles; reuse the same conv weights each cycle
        f = None
        for _ in range(self.cycles):
            f, h = self.forward_once(x, h)

        # classify using final cycle features
        z = f.view(f.size(0), -1)
        z = F.relu(self.fc1(z))
        logits = self.fc2(z)
        return logits