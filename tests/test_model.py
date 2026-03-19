import pytest
import torch
from nasnet import NASNet


class TestNASNetArchitecture:
    """Verify shapes, dtypes, and basic forward-pass behavior."""

    def test_output_shape(self, n_hidden, rng):
        model = NASNet(n_input=52, n_hidden=n_hidden)
        x = torch.randn(16, 52, generator=rng)
        logits = model(x)
        assert logits.shape == (16, 1)

    def test_single_waveform(self, rng):
        model = NASNet()
        x = torch.randn(1, 52, generator=rng)
        logits = model(x)
        assert logits.shape == (1, 1)

    def test_output_dtype_float32(self, rng):
        model = NASNet()
        x = torch.randn(8, 52, generator=rng)
        assert model(x).dtype == torch.float32

    def test_zero_input_not_nan(self):
        model = NASNet()
        x = torch.zeros(4, 52)
        logits = model(x)
        assert not torch.isnan(logits).any()


class TestClassify:
    """Verify classify() thresholding logic."""

    def test_classify_returns_bool(self, rng):
        model = NASNet()
        x = torch.randn(16, 52, generator=rng)
        result = model.classify(x, gamma=0.5)
        assert result.dtype == torch.bool
        assert result.shape == (16,)

    def test_classify_shape_matches_batch(self, rng):
        model = NASNet()
        for n in [1, 10, 100]:
            x = torch.randn(n, 52, generator=rng)
            assert model.classify(x).shape == (n,)

    def test_gamma_05_equivalent_to_logit_gt_zero(self, rng):
        """gamma=0.5 ↔ logit(0.5)=0, so classify should match logit > 0."""
        model = NASNet()
        x = torch.randn(64, 52, generator=rng)
        logits = model(x).squeeze(-1)
        expected = logits > 0.0
        actual = model.classify(x, gamma=0.5)
        assert torch.equal(expected, actual)

    def test_higher_gamma_fewer_spikes(self, rng):
        """Raising gamma should accept fewer (or equal) waveforms as spikes."""
        model = NASNet()
        x = torch.randn(200, 52, generator=rng)
        n_low = model.classify(x, gamma=0.3).sum()
        n_high = model.classify(x, gamma=0.7).sum()
        assert n_high <= n_low

    def test_gamma_extreme_values(self, rng):
        """gamma near 0 → almost all spikes; gamma near 1 → almost none."""
        model = NASNet()
        x = torch.randn(100, 52, generator=rng)
        assert model.classify(x, gamma=0.001).sum() >= model.classify(x, gamma=0.999).sum()


class TestMATLABEquivalence:
    """Verify the PyTorch forward pass matches the exact MATLAB math."""

    def test_manual_forward_matches_module(self, n_hidden, rng):
        """Compute the forward pass manually with matrix ops and compare."""
        model = NASNet(n_input=52, n_hidden=n_hidden)
        x = torch.randn(32, 52, generator=rng)

        # Manual computation matching MATLAB's runNASNetContinuous.m:
        #   hidden = relu(x @ W1^T + b1)
        #   logit  = hidden @ W2^T + b2
        with torch.no_grad():
            W1 = model.hidden.weight  # (H, 52)
            b1 = model.hidden.bias    # (H,)
            W2 = model.output.weight  # (1, H)
            b2 = model.output.bias    # (1,)

            hidden = torch.relu(x @ W1.T + b1)
            logit_manual = hidden @ W2.T + b2

        logit_module = model(x)
        assert torch.allclose(logit_manual, logit_module, atol=1e-6)
