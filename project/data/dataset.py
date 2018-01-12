import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Dynamic model ------------------
class ToTensorSeq(object):
    def __call__(self, sample):
        rgb, state, action, rgb_target, state_target = sample.values()
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


class DynamicDataSet(Dataset):
    '''Dataset for dynamics model
    Arguments:
        data:       list of data. [[rgb], [state], [action]]
        seq_len:    Sequence length (Integer) (Default=3)
    '''
    def __init__(self,
                 data,
                 seq_len=1,
                 img_shape=[40, 40, 3],
                 state_shape=[11],
                 action_shape=[2],
                 transform=ToTensorSeq()):
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
        sample = {'rgb': rgb,
                  'state': state,
                  'action': action,
                  'rgb_target': rgb_target,
                  'state_target': state_target}
        if self.transform:
            sample = self.transform(sample)
        return sample
# --------------------------------



class ToTensor(object):
    def __call__(self, obs, state):
        obs = obs.transpose((2, 0, 1))  #swap color axis np(seq,H,W,C) -> torch(seq,C,H,W)
        obs = torch.from_numpy(obs).float()
        obs /= 255.  # normalize
        state = torch.from_numpy(state).float()
        return obs, state


class ToTensorReacherPlane(object):
    def __call__(self, obs, state):
        obs = obs.transpose((2, 0, 1))  #swap color axis np(seq,H,W,C) -> torch(seq,C,H,W)
        obs = torch.from_numpy(obs).float()
        obs /= 255.  # normalize
        state = torch.from_numpy(state[:-2]).float()  # remove joint_speed
        return obs, state


class ReacherPlaneDataset(Dataset):
    '''Dataset for Understandigng model
    Translation:    input=RGB -> target=State
    Arguments:
        data:       {'obs': [obs_list], 'states': [states]}
    '''
    def __init__(self, data, transform=ToTensorReacherPlane()):
        self.obs = data['obs']
        self.state = data['states']
        self.transform = transform
        self.obs_shape = self.obs[0].shape
        self.state_shape = self.state[0].shape

    def __len__(self):  # prediction data, next frame, AFTER sequence is target.
        return len(self.obs)

    def __getitem__(self, idx):
        obs, state = self.obs[idx], self.state[idx]
        if self.transform:
            obs, state = self.transform(obs, state)
        return obs, state


def load_reacherplane_data(path, batch_size=300, num_workers=4, shuffle=True):
    '''Dataset for Understanding model ReacherPlane
    (joints_speeds are omitted)
    :param path        : path to data
    :param batch_size  : int
    :param num_workers : int
    :param shuffle     : Boolean
    '''
    data = torch.load(path)
    dset = ReacherPlaneDataset(data, ToTensorReacherPlane())
    tloader = DataLoader(dset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle)
    return dset, tloader

def load_dataset(path, batch_size=256, num_workers=4, shuffle=True, transform=ToTensor()):
    '''Dataset for Understanding model
    :param path        : path to data
    :param batch_size  : int
    :param num_workers : int
    :param shuffle     : Boolean
    :param transform   : transformation function
    '''
    data = torch.load(path)
    dset = ProjectDataSet(data, transform)
    tloader = DataLoader(dset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle)
    return dset, tloader


# Social ------------------
class Social_Dataset_numpy(Dataset):
    '''Dataset for trajectories for the Social environment class

    No transformation. Data is collected from env and stored as is.

    Arguments:
        data:       {'obs': [obs_list], 'states': [states_list]}
    '''
    def __init__(self, data):
        assert type(data) is dict
        self.obs = data['obs']
        self.state = data['states']
        self.transform = None
        self.obs_shape = self.obs[0].shape
        self.state_shape = self.state[0].shape

    def __len__(self):  # prediction data, next frame, AFTER sequence is target.
        return len(self.obs)

    def __getitem__(self, idx):
        return self.state[idx], self.obs[idx]
