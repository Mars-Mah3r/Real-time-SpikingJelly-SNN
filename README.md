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

***Alternatively, a directory can be set using the "Trained_checkpoints", to avoid training, and initialise trained weights automatically to the neural network***

## Evaluate through forward propogation inference. 

Have the event data from the DVS camera processed using the DV software and the ****camera_capture.py****

Ensure the save directory of the DVS_camera.npz is outputted to a directory that is concordant to the (1) below

![Tut](https://github.com/Mars-Mah3r/Real-time-SpikingJelly-SNN/assets/108829389/4b221c92-6ec4-47e5-b5be-45283676756c)

Frame data will be generated from the event_matrices_np.npz and saved in location (2). Ensure that recordings are kept short as large frame data generataion can cause too much GPU memory overhead<sup>1</sup> 

The .npz saved in location (3) should be correclty pre proccessed to fed into the trained SpikingJelly Network and produce a gesture classification label









<sup>1</sup>![Screenshot 2023-07-11 200757](https://github.com/Mars-Mah3r/Real-time-SpikingJelly-SNN/assets/108829389/d3f6cfa2-347e-46bb-b029-4cb1970dc1ae) 





