# Gesture Training


# Installation

## prerequisites
* Python 3.5
* Pytorch 0.2.0
* CUDA 8.0 (if GPU)
* [Gym](https://github.com/openai/gym)
* [Roboschool](https://github.com/openai/roboschool)

## Setup

1. Create a Conda environment (python=3.5)
2. Source environment
3. Install opencv
```bash
conda install -c menpo opencv3
```
4. Install PyTorch
```bash
conda install pytorch torchvision -c pytorch
``` 
5. cd into your roboschool folder and pip install
```bash
cd $ROBOSCHOOL_PATH
pip install -e .
``` 
6. Clone this repo 
```bash
git clone https://github.com/ErikEkstedt/Gestures
cd Gestures
pip install -e .
```


