# ROBOT
## Main

* Loads data from file, initializes dataset, model and optizer and trains for an arbitrary number of epochs, then visualizes the result. Saves result images after each epoch.

## main_test.py

Should be a testing version of main that does note use argparse for arguments.

## model.py

The model script that contains the actual model.

### model_new.py

Were I try out the new model structures. 

Todo :
model can't make rollout prediction. 



## CLSTMCell - Convolutional LSTM

* Recurrent network architectures such as LSTM are well suited for time series prediction because of their inherent temporal information processing.

* Convolutional network architectures are well suited for image data where the information is highly spatially correlated.

In order to handle data which exhibits both spatial and temporal dependencies we want to take the best out of two worlds and combine an LSTM with Convolutions. This was done in the paper [Convolutional (LSTM) Network: (A) Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214 "Arxiv: paper") by Shi et al and then used in the inspiration for this project in [Deep Visual Foresight for Planning Robot Motion](http://arxiv.org/abs/1610.00696) by Finn and Levine.

## Environment
* Environment used: Mujoco Reacher-v0
* This script was to experiment with camera angle, and env.render('rgb_array') customization.

## RobotDataset
* THis dataset looks in a directory and loads a file (currently just the first it sees) into a pytorch dataset. This may then be used by pytorch's DataLoader.

## Collect_data
* Collects data by taking random actions in one long trajectory (Reacher does have an intrinsic stop signal)

## Logger
* A slightly tweaked logger class from [logger.py from yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py "Github: yunjeu-logger") that uses tensorboard to log values, images and histagram over network parameters.
* Creates directories as $ROOT/run-x, $ROOT/run-x/logs, $ROOT/run-x/checkpoints. 
* Saves checkpoints with the models and optimizers state_dict and what epoch the training is at and the loss.
