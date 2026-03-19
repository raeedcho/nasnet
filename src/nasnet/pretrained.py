import torch
from pathlib import Path
from .model import NASNet

_WEIGHTS_DIR = Path(__file__).parent / "weights"

AVAILABLE_NETWORKS = [
    "MotorNet",
    "UberNet_N50_L1",
    "uStimNet",
    "uStimNet_artifact",
]


def load_pretrained(
    network_name: str = "UberNet_N50_L1",
    device: str = "cpu",
) -> NASNet:
    """Load a pretrained NASNet by name.

    Args:
        network_name: one of AVAILABLE_NETWORKS
        device: torch device string
    Returns:
        NASNet model with pretrained weights, in eval mode
    Raises:
        ValueError: if network_name is not recognized
        FileNotFoundError: if the .pt file is missing
    """
    if network_name not in AVAILABLE_NETWORKS:
        raise ValueError(
            f"Unknown network '{network_name}'. "
            f"Available: {AVAILABLE_NETWORKS}"
        )

    weights_path = _WEIGHTS_DIR / f"{network_name}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Infer architecture dimensions from the saved state dict
    n_hidden, n_input = state_dict["hidden.weight"].shape  # (H, 52)

    model = NASNet(n_input=n_input, n_hidden=n_hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model
