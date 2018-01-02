import pickle
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from project.dynamics_model.robotdataset import RobotDataset
from project.dynamics_model.logger import Logger
from project.dynamics_model.new_model import Model

# hyperparameters
seq_len = 3
batch = 32
nworkers = 4

# Save path and logger
resultpath = "/home/erik/DATA/Project"
logger = Logger(resultpath)

# Desktop path
trainfile = "/home/erik/DATA/Project/data/data0.p"
data = pickle.load(open(trainfile, "rb"))
dset = RobotDataset(data, seq_len)
tloader = DataLoader(dset, batch_size=batch, num_workers=nworkers, shuffle=True)

valfile = "/home/erik/DATA/Project/data/data1.p"
data = pickle.load(open(valfile, "rb"))
dset = RobotDataset(data, seq_len)
vloader = DataLoader(dset, batch_size=batch, num_workers=nworkers, shuffle=True)

# Model
# Cuda = torch.cuda.is_available()
Cuda = False
#print('Cuda set to: ', Cuda)
input_size = (3, 40, 40)
model = Model(input_size, CUDA=Cuda)
if Cuda:
    model = model.cuda()

opt = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.MSELoss()


# TESTING
dataiter = iter(tloader)
data = next(dataiter)

o = Variable(data['rgb'])
a = Variable(data['action'])
s = Variable(data['state'])
o_ = Variable(data['rgb_target'], requires_grad=False)
s_ = Variable(data['state_target'], requires_grad=False)

rgb_out, s_out, _ = model(o, action=a, state=s)

print('rgb out: ', rgb_out.size())
print('state out: ', s_out.size())
