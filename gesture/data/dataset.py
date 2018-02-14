import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    def __call__(self, state, obs):
        obs = obs.transpose((2, 0, 1))  #swap color axis np(seq,H,W,C) -> torch(seq,C,H,W)
        obs = torch.from_numpy(obs).float()
        obs /= 255.  # normalize
        return torch.from_numpy(state).float(), obs


class UnderstandDataset(Dataset):
    '''Dataset for Understandigng model
    Arguments:
        :data       : Dict: {'obs': [obs_list], 'states': [states]}
    '''
    def __init__(self, data, transform=ToTensor()):
        self.obs = data['obs']
        self.state = data['states']
        self.transform = transform
        self.obs_shape = self.obs[0].shape
        self.state_shape = self.state[0].shape

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.transform(self.state[idx], self.obs[idx])

class UnderstandDatasetCuda(Dataset):
    '''Dataset for Understandigng model
    Arguments:
        :data       : Dict: {'obs': [obs_list], 'states': [states]}
    '''
    def __init__(self, data):
        self.obs = data['obs']
        self.state = data['states']
        self.obs_shape = self.obs[0].shape
        self.state_shape = self.state[0].shape
        # self.transform_to_cuda()

    def __len__(self):
        return len(self.obs)

    def transform_to_cuda(self, vel=2):
        obs_cuda, state_cuda = [], []
        for s, obs in tqdm(zip(self.state, self.obs)):
            obs = obs.transpose((2, 0, 1))  #swap color axis np(seq,H,W,C) -> torch(seq,C,H,W)
            obs = torch.from_numpy(obs).float()
            obs /= 255.  # normalize
            obs = obs.cuda()
            s = torch.from_numpy(s[:-vel]).float().cuda()
            obs_cuda.append(obs)
            state_cuda.append(s)
        self.obs = obs_cuda
        self.state = state_cuda

    def __getitem__(self, idx):
        return self.state[idx], self.obs[idx]
