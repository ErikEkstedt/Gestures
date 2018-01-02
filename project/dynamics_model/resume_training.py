import torch
from torch.autograd import Variable
import numpy as np
from new_model import Model
from robotdataset import get_data
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# Hyperparameters
seq_len = 3
batch = 8
nworkers = 4
epochs = 500

# Load data
trainfile = "/home/erik/DATA/Project/data/data0.p"
dset, tloader = get_data(trainfile)

# model
input_size = (3, 40, 40)
model = Model(input_size)


# Load training checkpoint
checkpoint_path = "/home/erik/DATA/Project/Oct6/run-0/checkpoints/checkpoint.pth.tar"
# checkpoint_path = "/home/erik/DATA/Project/checkpoints_desktop/checkpoint.pth.tar"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

# checkpoint, start_epoch, best_prec1, model_state_dict, optimizer_state_dict = load_checkpoint(checkpoint_path) # broken?

# playing around
rgbs, states, actions = dset.data
dataiter = iter(tloader)
data = next(dataiter)
x = Variable(data['rgb'])
a = Variable(data['action']).float()
s = Variable(data['state']).float()
s_target = Variable(data['state_target'], requires_grad=False).float()
x_target = Variable(data['rgb_target'], requires_grad=False)


def print_future(future):
    rgb = future['rgb']
    # only visualize 1 batch
    img_list = []
    for batch in rgb:
        img_list.append(batch[0])
    show_tensor(make_grid(img_list))

model.reset_hidden()
steps = 5
a_next = torch.zeros(steps, batch, 1, 2)
future = model.predict_future(model, rgb=x, state=s, action=a, next_actions=a_next, step=steps, show=True)
print_future(future)
