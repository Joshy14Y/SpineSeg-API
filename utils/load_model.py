import torch
from pathlib import Path
from models.u_net import UNet


def load_model(weights_path: Path, device: str):
    """Load UNet model from weights and set to evaluation mode."""
    model = UNet(base_channels=16, out_channels=18)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
