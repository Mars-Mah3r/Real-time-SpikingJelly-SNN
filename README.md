# Real-time-SpikingJelly-SNN

## Introduction
This repository is based of the [Spikingjelly](https://github.com/fangwei123456/spikingjelly) by [fangwei123456](https://github.com/fangwei123456), refer to the original documentation for further notes. 

The aim of this repository is to use the SpikingJelly spike-neural-network model is to be able to recieve as an input, real time data from a DVS camera, and use the SpikingJelly model to train the data in real time. 

## Requirements 
```
torch
matplotlib
numpy
tqdm
torchvision
scipy
```

**Refer to original GitHub, for Device support**
## Installation
(from [Spikingjelly](https://github.com/fangwei123456/spikingjelly))

### Install the last stable version from [PyPI](https://pypi.org/project/spikingjelly/):
```
pip install spikingjelly
```
### Install the latest developing version from the source codes, from [GitHub](https://github.com/fangwei123456/spikingjelly)
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```

## Unpacking the DVS Gesture data
(note: this current implementation is training using the [DVS128 gesture dataset from IBM](https://research.ibm.com/interactive/dvsgesture/), later iterations of this repo will incorporate a method of implentening real time data training.)

