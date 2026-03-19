import torch
import torch.nn as nn


class NASNet(nn.Module):
    """Single hidden-layer ReLU network for spike/noise classification.

    Architecture (from Issar et al. 2020):
        Input [N, n_input] → Linear(n_input, n_hidden) + ReLU → Linear(n_hidden, 1) → logit

    Forward pass returns raw logits. Use classify() for binary decisions.
    """

    def __init__(self, n_input: int = 52, n_hidden: int = 50):
        super().__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: waveforms, shape (N, n_input), dtype float32
        Returns:
            logits, shape (N, 1)
        """
        return self.output(torch.relu(self.hidden(x)))

    def classify(self, x: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
        """Return boolean mask: True = spike, False = noise.

        Equivalent to sigmoid(logit) > gamma, but avoids the sigmoid call
        by comparing the raw logit to log(gamma / (1 - gamma)).

        Args:
            x:     waveforms, shape (N, n_input)
            gamma: threshold in (0, 1); higher = more selective
        Returns:
            is_spike: bool tensor, shape (N,)
        """
        logit_threshold = torch.log(torch.tensor(gamma / (1.0 - gamma)))
        logits = self.forward(x).squeeze(-1)
        return logits > logit_threshold
