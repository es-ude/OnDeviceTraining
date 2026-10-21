"""The HAR and mnist_cnn PyTorch twins are expressed with modules only."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.fx
import torch.nn.functional as F

from examples.har_classifier import train_pytorch as har
from examples.mnist_cnn import train_pytorch as mnist


def _reference_forward(m, x, gap_kernel):
    """The functional forward the twins had before they were rewritten with modules."""
    x = F.relu(m.conv1(x))
    x = F.max_pool1d(x, 2)
    x = F.relu(m.conv2(x))
    x = F.max_pool1d(x, 2)
    if hasattr(m, "conv3"):
        x = F.relu(m.conv3(x))
    x = F.avg_pool1d(x, gap_kernel)
    x = x.flatten(1)
    return m.fc(x)


def test_har_forward_equals_functional_reference():
    torch.manual_seed(0)
    m = har.HarModel()
    x = torch.randn(4, 9, 128)
    assert torch.equal(m(x), _reference_forward(m, x, 32))


def test_mnist_forward_equals_functional_reference():
    torch.manual_seed(0)
    m = mnist.MnistCnn()
    x = torch.randn(4, 1, 784)
    assert torch.equal(m(x), _reference_forward(m, x, 196))


def test_twin_state_dict_keys_unchanged():
    har_keys = {f"{layer}.{p}" for layer in ("conv1", "conv2", "conv3", "fc") for p in ("weight", "bias")}
    assert set(har.HarModel().state_dict()) == har_keys
    mnist_keys = {k for k in har_keys if not k.startswith("conv3.")}
    assert set(mnist.MnistCnn().state_dict()) == mnist_keys


def test_twins_trace_to_module_calls_only():
    for m in (har.HarModel(), mnist.MnistCnn()):
        ops = {n.op for n in torch.fx.symbolic_trace(m).graph.nodes}
        assert ops == {"placeholder", "call_module", "output"}


def test_mnist_to_model_input_shape():
    for arr in (np.zeros((3, 28, 28)), np.zeros((3, 784))):
        out = mnist.to_model_input(arr)
        assert tuple(out.shape) == (3, 1, 784)
        assert out.dtype == torch.float32
