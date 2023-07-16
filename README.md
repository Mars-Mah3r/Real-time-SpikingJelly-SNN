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
<sub>Other dependancies such as DV etc may be needed for pre-processing elements</sub>
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





## 1. Unpacking the DVS Gesture data 
### _( SpikeJelly_Process_DVS.py )_
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






## 2. Training the Network with IBM's dataset 

Open the _SpikeJelly_classify_DVS.py_ and run the following command
```
python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8
```
making sure to replace the "/datasets/DVSGesture", with the same directory you have used earlier.

***Alternatively, a directory can be set using the "Trained_checkpoints", to avoid training, and initialise trained weights automatically to the neural network***


## 3. Custom Pre-processing

It should note that processing the Neurophormic data into a fornat that can be passed into the trained network for inference is termpermental. None the less here are the following files that will will aid in this process. (some things will need to be tweaked in this pipeline)

#### 3.1 camera_capture.py
This file will take a stream of data from the camera and convert the event stream into a .npz in the shape of 
```
( t, x, y, p )
```

#### 3.2 events_to_frame.py
This code has been adapted from the original _Classify_DVS.py_ (from [Spikingjelly](https://github.com/fangwei123456/spikingjelly)). 
This will convert the t, x, y, p data into a form being (20, 2, 128, 128)
```
20 => the amount of frames
2 => chanel, configured to 2 for RGB, 1 for greyscale
128, 128 => dimension of data
```

#### 3.3 Integrating data.
The data at times can be extremely large to process locally, and this file can convert the t x y p data into a more manageable format. More details can be found [here](https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.14/activation_based_en/neuromorphic_datasets.html#fixed-duration-integrating
)





## 4. Evaluate through forward propogation inference. 

Have the event data from the DVS camera processed using the DV software and the ****camera_capture.py****

Ensure the save directory of the DVS_camera.npz is outputted to a directory that is concordant to the (1) below

![Tut](https://github.com/Mars-Mah3r/Real-time-SpikingJelly-SNN/assets/108829389/4b221c92-6ec4-47e5-b5be-45283676756c)

Frame data will be generated from the event_matrices_np.npz and saved in location (2). Ensure that recordings are kept short as large frame data generataion can cause too much GPU memory overhead<sup>1</sup> 

The .npz saved in location (3) should be correclty pre proccessed to fed into the trained SpikingJelly Network and produce a gesture classification label

There are several evaluation methods that have been tested, their feasibility are listed below ind desecending order: 

#### -- 4.1 SpikeJelly_Eval.py
Prints output tensors of all 0

#### -- 4.2 SpikeJelly_Eval_V1.7.py
A previous iteration 

#### -- 4.3 SpikeJelly_Eval_V1.6_reduced data usage.py
An attempt to circumnavigate high GPU memory requirements

#### Misc
-- 4.x check integrated frame shape.py

-- 4.x SpikeJelly_Eval_of_test_ibm_data.p
This correctly can take a single .npz from the IBM dataset and classify the data correctly. 







<sup>1</sup>![Screenshot 2023-07-11 200757](https://github.com/Mars-Mah3r/Real-time-SpikingJelly-SNN/assets/108829389/d3f6cfa2-347e-46bb-b029-4cb1970dc1ae) 





