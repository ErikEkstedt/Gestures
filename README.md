# Gesture Training


# Installation

## prerequisites
* Conda
* Python 3.5
* Pytorch 0.2.0
* CUDA 8.0 (if GPU)
* [Gym](https://github.com/openai/gym)
* [Roboschool](https://github.com/openai/roboschool)

## Setup
This has been tested on Ubuntu 16.04

1. Create a Conda environment (python=3.5)
```bash
conda create -n gesture python=3.5
```
2. Source environment
```bash
source activate gesture
```
4. Install PyTorch
```bash
conda install pytorch torchvision -c pytorch
```
3. Install opencv
Opencv is used to render images with cv2.imshow. This package is large, might be a better way but this works.
```bash
conda install -c conda-forge opencv
``` 
5. cd into your roboschool folder and pip install
```bash
cd $ROBOSCHOOL_PATH
pip install -e .
``` 
6. Clone this repo and install
```bash
git clone https://github.com/ErikEkstedt/Gestures
cd Gestures
pip install -e .
```
7. Install requirements
```bash
pip install -r requirements.txt
```



