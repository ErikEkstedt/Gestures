import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader


class ToTensor(object):
    def __call__(self, sample):
        rgb, state = sample['rgb'], sample['state']
        action, rgb_target = sample['action'], sample['rgb_target']
        state_target = sample['state_target']
        dims = rgb[0].shape
        if dims[1] is not 3:
            # swap color axis. numpy: (seq,H,W,C) -> torch: (seq,C,H,W)
            rgb_target = rgb_target.transpose((2, 0, 1))  # transpose target
            r = []
            for i in range(len(rgb)):
                # transpose sequence
                r.append(rgb[i].transpose((2, 0, 1)))
        else:
            r = rgb

        rgb = np.stack(r)
        rgb, rgb_target = torch.from_numpy(rgb).float(), torch.from_numpy(rgb_target).float()
        rgb /= 255.
        rgb_target /= 255.
        state, action = np.stack(state), np.stack(action)
        return {'rgb': rgb,
                'state': torch.from_numpy(state),
                'action': torch.from_numpy(action),
                'rgb_target': rgb_target,
                'state_target': torch.from_numpy(state_target)}


class RobotDataset(Dataset):
    '''Dataset for Robot Project
    Arguments:
        data:       list of data

        seq_len:    Sequence length (Integer) (Default=3)
    '''
    def __init__(self, data, seq_len=1, img_shape=[40, 40, 3], state_shape=[11], action_shape=[2], transform=ToTensor()):
        assert data is not None
        self.data = data
        self.transform = transform

        self.seq_len = seq_len
        self.img_shape = [seq_len] + img_shape
        self.state_shape = [seq_len] + state_shape
        self.action_shape = [seq_len] + action_shape

    def __len__(self):  # prediction data, next frame, AFTER sequence is target.
        return len(self.data[0]) - self.seq_len - 1

    def __getitem__(self, idx):
        rgb, state, action = [], [], []
        for i in range(self.seq_len):
            rgb.append(self.data[0][idx + i])
            state.append(self.data[1][idx + i])
            action.append(self.data[2][idx + i])

        rgb_target = self.data[0][idx + self.seq_len]
        state_target = self.data[1][idx + self.seq_len]
        sample = {'rgb': rgb, 'state': state, 'action': action, 'rgb_target': rgb_target, 'state_target': state_target}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_data(batch_size, seq_len=1, num_workers=4, shuffle=True, dpath=None):
    assert batch_size > 0
    if dpath is None:
        dpath = "/home/erik/DATA/Project/data/data0.p"

    data = pickle.load(open(dpath, "rb"))
    dset = RobotDataset(data)
    tloader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dset, tloader


class ToTensor_Baseline(object):
    def __call__(self, sample):
        rgb, state = sample['rgb'], sample['state']
        rgb = rgb.transpose((2, 0, 1))  # transpose target
        rgb = torch.from_numpy(rgb).float()
        rgb /= 255.
        return {'rgb': rgb, 'state': torch.from_numpy(state).float()}


class Dataset_Baseline(Dataset):
    '''Dataset for Robot Project
    Arguments:
        data:       list of data
    '''
    def __init__(self, data, img_shape=[40, 40, 3], state_shape=[11], transform=ToTensor_Baseline()):
        assert data is not None
        self.data = [data[0], data[1]]  # Don't need all data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        rgb = self.data[0][idx]
        state = self.data[1][idx]  # Index 6,7 contains non-relevant data
        index = np.array([0, 1, 2, 3])  # only the fours firstvalues
        state = state[index]
        state = np.concatenate((state[:6], state[8:]))  # Discard non-relevant data
        sample = {'rgb': rgb, 'state': state}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_data_baseline(batch_size, num_workers=4, shuffle=True, dpath=None):
    assert batch_size > 0
    if dpath is None:
        dpath = "/home/erik/DATA/Project/data/data0.p"
    data = pickle.load(open(dpath, "rb"))
    dset = Dataset_Baseline(data)
    dloader = DataLoader(dset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dset, dloader
