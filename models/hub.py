# Defines the hub state. Takes in summaries and updates global state
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hub(nn.Module):
    """
    A simple 'global state' module.
    Takes a feature summary vector s_t and updates a hidden hub state h_t.
    """
    def __init__(self, feat_dim: int, hub_dim: int):
        super().__init__()
        self.hub_dim = hub_dim

        # update rule: h_{t+1} = tanh(W_s s_t + W_h h_t)
        self.Ws = nn.Linear(feat_dim, hub_dim)
        self.Wh = nn.Linear(hub_dim, hub_dim)

    def init_state(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.hub_dim, device=device)

    def forward(self, s_t, h_t):
        # s_t: (B, feat_dim)
        # h_t: (B, hub_dim)
        h_next = torch.tanh(self.Ws(s_t) + self.Wh(h_t))
        return h_next