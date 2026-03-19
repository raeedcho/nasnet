import pytest
import torch
from nasnet import NASNet, load_pretrained, AVAILABLE_NETWORKS


class TestLoadPretrained:
    """Test loading every bundled pretrained network."""

    def test_load_returns_nasnet(self, network_name):
        model = load_pretrained(network_name)
        assert isinstance(model, NASNet)

    def test_model_is_eval_mode(self, network_name):
        model = load_pretrained(network_name)
        assert not model.training

    def test_input_dim_is_52(self, network_name):
        model = load_pretrained(network_name)
        assert model.hidden.in_features == 52

    def test_output_dim_is_1(self, network_name):
        model = load_pretrained(network_name)
        assert model.output.out_features == 1

    def test_forward_pass_runs(self, network_name, rng):
        model = load_pretrained(network_name)
        x = torch.randn(8, 52, generator=rng)
        logits = model(x)
        assert logits.shape == (8, 1)
        assert not torch.isnan(logits).any()

    def test_classify_runs(self, network_name, rng):
        model = load_pretrained(network_name)
        x = torch.randn(8, 52, generator=rng)
        result = model.classify(x, gamma=0.5)
        assert result.shape == (8,)
        assert result.dtype == torch.bool

    def test_weights_are_not_default_init(self, network_name):
        """Loaded weights should differ from a freshly initialized model."""
        loaded = load_pretrained(network_name)
        fresh = NASNet(
            n_input=loaded.hidden.in_features,
            n_hidden=loaded.hidden.out_features,
        )
        # It is astronomically unlikely that random init matches trained weights
        assert not torch.equal(loaded.hidden.weight, fresh.hidden.weight)

    def test_deterministic_output(self, network_name, rng):
        """Same input → identical output across two loads."""
        x = torch.randn(16, 52, generator=rng)
        m1 = load_pretrained(network_name)
        m2 = load_pretrained(network_name)
        assert torch.equal(m1(x), m2(x))


class TestLoadErrors:
    """Test error handling for invalid inputs."""

    def test_unknown_network_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown network"):
            load_pretrained("NotARealNetwork")

    def test_available_networks_is_nonempty(self):
        assert len(AVAILABLE_NETWORKS) == 4

    def test_device_cpu(self, network_name):
        model = load_pretrained(network_name, device="cpu")
        param = next(model.parameters())
        assert param.device == torch.device("cpu")
