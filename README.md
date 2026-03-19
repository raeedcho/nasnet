# nasnet

PyTorch implementation of NASNet spike sorter (Issar et al. 2020, *J Neurophysiol*).

## Installation

```bash
pip install git+https://github.com/raeedcho/nasnet.git
```

## Usage

```python
from nasnet import load_pretrained

model = load_pretrained("UberNet_N50_L1")
is_spike = model.classify(waveforms, gamma=0.5)
```

## Available Networks

There are a few pre-trained networks, imported from [the MATLAB implementation of NASnet](https://github.com/SmithLabNeuro/nasnet)
- `MotorNet`
- `UberNet_N50_L1`
- `uStimNet`
- `uStimNet_artifact`

## Reference

Issar D, Bhagat SM, Smith MA, Bhatt S. 2020. Neural artifact sorting with the
Neural Artifact Sorter Network (NASNet). *J Neurophysiol* 124(3):845–856.
https://doi.org/10.1152/jn.00641.2019
