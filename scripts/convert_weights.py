"""Download NASNet MATLAB weights and convert to PyTorch .pt state dicts.

Usage:
    python scripts/convert_weights.py
"""
import urllib.request
import numpy as np
import torch
import pathlib

from nasnet.model import NASNet

BASE_URL = "https://raw.githubusercontent.com/SmithLabNeuro/nasnet/main/networks"
NETWORKS = ["MotorNet", "UberNet_N50_L1", "uStimNet", "uStimNet_artifact"]
SUFFIXES = ["_w_hidden", "_b_hidden", "_w_output", "_b_output"]
OUT_DIR = pathlib.Path("src/nasnet/weights")


def download_text_weights(dest_dir: pathlib.Path) -> None:
    """Download all four weight text files for every network."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in NETWORKS:
        for suffix in SUFFIXES:
            filename = f"{name}{suffix}"
            url = f"{BASE_URL}/{filename}"
            out_path = dest_dir / filename
            if not out_path.exists():
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, out_path)


def convert_one(network_name: str, text_dir: pathlib.Path) -> dict:
    """Load MATLAB text weights and return a PyTorch state_dict.

    Transposition rules (MATLAB → PyTorch nn.Linear):
      - W1 text file: shape (n_input, H) → transpose to (H, n_input) for hidden.weight
      - b1 text file: shape (H,) → no change for hidden.bias
      - W2 text file: shape (H,) → reshape to (1, H) for output.weight
      - b2 text file: scalar → reshape to (1,) for output.bias
    """
    w1 = np.loadtxt(text_dir / f"{network_name}_w_hidden")   # (52, H)
    b1 = np.loadtxt(text_dir / f"{network_name}_b_hidden")   # (H,)
    w2 = np.loadtxt(text_dir / f"{network_name}_w_output")   # (H,)
    b2 = np.loadtxt(text_dir / f"{network_name}_b_output")   # scalar

    return {
        "hidden.weight": torch.tensor(w1.T, dtype=torch.float32),            # (H, 52)
        "hidden.bias":   torch.tensor(b1, dtype=torch.float32),              # (H,)
        "output.weight": torch.tensor(w2[np.newaxis], dtype=torch.float32),  # (1, H)
        "output.bias":   torch.tensor([float(b2)], dtype=torch.float32),     # (1,)
    }


def main():
    tmp_dir = pathlib.Path("_tmp_matlab_weights")
    download_text_weights(tmp_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in NETWORKS:
        sd = convert_one(name, tmp_dir)
        out_path = OUT_DIR / f"{name}.pt"
        torch.save(sd, out_path)

        H, n_in = sd["hidden.weight"].shape
        print(f"Saved {out_path}  (n_input={n_in}, n_hidden={H})")

    # Verify round-trip for every network
    for name in NETWORKS:
        sd = torch.load(OUT_DIR / f"{name}.pt", weights_only=True)
        H, n_in = sd["hidden.weight"].shape
        model = NASNet(n_input=n_in, n_hidden=H)
        model.load_state_dict(sd)
        print(f"Verified {name}: load_state_dict succeeded")


if __name__ == "__main__":
    main()
