import pytest
import torch
from nasnet import NASNet, AVAILABLE_NETWORKS, load_pretrained


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture(params=[10, 50, 128])
def n_hidden(request):
    return request.param


@pytest.fixture(params=AVAILABLE_NETWORKS)
def network_name(request):
    return request.param
