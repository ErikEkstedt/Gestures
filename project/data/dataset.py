import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    def __call__(self, obs, state):
        obs = obs.transpose((2, 0, 1))  #swap color axis np(seq,H,W,C) -> torch(seq,C,H,W)
        obs = torch.from_numpy(obs).float()
        obs /= 255.  # normalize
        return obs, torch.from_numpy(state).float()


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
        return self.transform(self.obs[idx], self.state[idx])
