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
_Refer to original GitHub, for Device support_

## Installation
(from [Spikingjelly](https://github.com/fangwei123456/spikingjelly))

#### Install the last stable version from [PyPI](https://pypi.org/project/spikingjelly/):
```
pip install spikingjelly
```
#### Install the latest developing version from the source codes, from [GitHub](https://github.com/fangwei123456/spikingjelly)
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```

## Unpacking the DVS Gesture data
(note: this current implementation is training using the [DVS128 gesture dataset from IBM](https://research.ibm.com/interactive/dvsgesture/), later iterations of this repo will incorporate a method of implentening real time data training.)

Download the  [DVS128 gesture dataset from IBM](https://research.ibm.com/interactive/dvsgesture/), and save it to a particular directory e.g. _'D:/datasets/DVS128Gesture'_, the directory structure should be as follows: 
```
.
|-- DvsGesture.tar.gz
|-- LICENSE.txt
|-- README.txt
`-- gesture_mapping.csv
```

***Open the SpikeJelly_Process_DVS.py***, and on line 16, replace the directory with the directtory you have downloaded the dataset e.g.:
```
root_dir = 'D:/datasets/DVS128Gesture'
```
Now run the python scipt and wait for the DVS128 dataset to extract. 

## Classify the dataset 

Open the _SpikeJelly_classify_DVS.py_ and run the following command
```
python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8
```
making sure to replace the "/datasets/DVSGesture", with the same directory you have used earlier.


