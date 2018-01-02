import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def imshow_test(x, t, out):
    '''Turns torch.FloatTensors to numpy arrays, transposes them
    and plots them as images.

    Arguments:
        x, t, out - torch.FloatTensor

    '''
    x, t, out = x.numpy(), t.numpy(), out.numpy()
    x, t, out = x.transpose(), t.transpose(), out.transpose()

    plt.subplot(2,2,1)
    plt.imshow(x)
    plt.subplot(2,2,2)
    plt.imshow(t)
    plt.subplot(2,2,3)
    plt.imshow(x)
    plt.subplot(2,2,4)
    plt.imshow(out)
    plt.show()

def tensorshow(timg):
    img = timg[0].numpy()
    img = img.transpose(1,2,0)
    plt.imshow(img)
    plt.show()

def list2Loader(data, batch_size, shuffle=False, num_workers=1):
    '''
    #Arguments:
        data - list of torch.Tensors or list of lists of torch.Tensors
        batch_size - integer
        shuffle - boolean, shuffle data or not.
        num_workers - number of workers.
    #Output:
        torch.data.utils.DataLoader
    '''
    def list_tensor(data):
        ''' Converts a list of Tensors (data)/list of lists of tensors to a Tensor '''
        if type(data[0]) == list:
            T = list_tensor(data[0])
            for i in trange(1,len(data)):
                T = torch.cat((T,list_tensor(data[i])))
            return T
        else:
            tensor = data[0]
            for d in data:
                tensor = torch.cat((tensor,d),dim=0)
            return tensor
    tensor_data = list_tensor(data)
    # prediction models has the next data as target for the current.
    a = tensor_data[:-1]; b = tensor_data[1:]
    dset = TensorDataset(a,b)
    return dset, DataLoader(dset, batch_size = batch_size, shuffle=shuffle, num_workers=num_workers)

def train_val_test(data, p = [0.2,0.7,0.1]):
    '''
    #Arguments:
        data - list of data
        p - list of train/val/test partitions
    #Returns:
        3 lists of train/validation/test data
    '''
    def traindata(data):
        return data[:-1], data[1:]

    tr_list = []; val_list = []; te_list=[]
    unfold= []
    for l in data:
        for d in l:
            unfold.append(d)
    n = len(unfold)
    tr = int(n*p[0])
    te = int(n*p[1])
    va = int(n*p[2])
    for i in range(tr):
        tr_list.append(unfold[i])
    for i in range(tr,tr+va):
        val_list.append(unfold[i])
    for i in range(va,tr+va+te):
        te_list.append(unfold[i])
    return traindata(tr_list), traindata(val_list), traindata(te_list)

def gather_frames(env, num_frames, render=False, default_action=0):
    ''' Gathers num_frames frames from environment env
    #Arguments:
        env - gym environment gym.make('PongDeterministic-v4') is default
        num_frames - int, number of frames.
        default_action - action that the "agent" performs during entire rfam gathering
    #Returns:
        list of lists of episode frames (torch.FloatTensor)
    '''
    def prepro(I):
        """ prepro 210x160x3 uint8 frame into  (1,3,40,40) 4D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,:] # downsample by factor of 2
        I = I[::2,::2,:].astype('float32') # downsample by factor of 2.
        I /= 255.
        I = I.transpose(2,0,1)
        return torch.from_numpy(I[None,:]).float()

    data = []; data.append([])
    t = 0
    s = env.reset()
    for i in trange(num_frames):
        if render: env.render()
        s = prepro(s)
        data[t].append(s)
        s,_,done,_ = env.step(default_action)
        if done:
            data.append([])
            t+=1
            s = env.reset()
    return data

def mem_usage(frames, frame_shape=(210, 160,3), dtype='uint8'):
    ''' calculates memory usage for storing frames of a certain data type.
    default values are for 'PongDeterministic-v4' '''
    if dtype == 'uint8':
        n = 1 # 1 byte per element
    elif dtype == 'int64':
        n = 8 # 8 bytes per element
    else:
        print('Dont know dtype')
        return
    elem = 1
    for s in frame_shape:
        elem*=s
    return frames*elem/1e9 # returns gb

def len_episodes(data):
    ''' Prints length of lists in input: data (list). '''
    for i,ep in enumerate(data):
        print(' Ep {} length: {}'.format(i, len(ep)))

def Conv2d_out_shape(Conv, input_shape, verbose=False, batch=False):
    '''Output of nn.Conv2d.
    #Arguments:
        input_shape - shape of input. (N,C,H,W) or (C,H,W)
        Conv - nn.Conv2d()

    From PyTorch Documentation:
            http://pytorch.org/docs/master/nn.html#conv2d
    Assumes channel first (N,C,H,W) or (C,H,W)
    '''
    if len(input_shape) > 3:
        # contains batch dimension
        batch = True
        h_in = input_shape[2]
        w_in = input_shape[3]
    else:
        # no batch dimension
        h_in = input_shape[1]
        w_in = input_shape[2]
    s = Conv.stride
    k = Conv.kernel_size
    p = Conv.padding
    d = Conv.dilation
    if verbose:
        print('stride: ', s)
        print('kernel: ', k)
        print('padding: ', p)
        print('h_in: ', h_in)
        print('w_in: ', w_in)
    # from numpy import floor
    h = np.floor((h_in + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
    w = np.floor((w_in + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)

    return (input_shape[0], Conv.out_channels, h, w) if batch else (Conv.out_channels, h, w)

def ConvTranspose2d_out_shape(Conv, input_shape, verbose=False, batch=False):
    '''Output shape of nn.ConvTranspose2d.
    #Arguments:
        input_shape - shape of input. (N,C,H,W) or (C,H,W)
        Conv - nn.ConvTranspose2d()

    From PyTorch Documentation:
            http://pytorch.org/docs/master/nn.html#conv2d
    Assumes channel first (N,C,H,W) or (C,H,W)
    '''
    if len(input_shape) > 3:
        # contains batch dimension
        batch = True
        h_in = input_shape[2]
        w_in = input_shape[3]
    else:
        # no batch dimension
        h_in = input_shape[1]
        w_in = input_shape[2]

    s = Conv.stride
    k = Conv.kernel_size
    p = Conv.padding
    op = Conv.output_padding
    # d = Conv.dilation
    if verbose:
        print('stride: ', s)
        print('kernel: ', k)
        print('padding: ', p)
        print('h_in: ', h_in)
        print('w_in: ', w_in)
    h = (h_in - 1) * s[0] - 2 * p[0] + k[0] + op[0]
    w = (w_in - 1) * s[1] - 2 * p[1] + k[1] + op[1]
    return (input_shape[0], Conv.out_channels, h, w) if batch else (Conv.out_channels, h, w)

def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model_state_dict = checkpoint['state_dict']
        optimizer_state_dict = checkpoint['optimizer']
        print("=> loaded checkpoint '{}' (epoch {})" .format(checkpoint_path, checkpoint['epoch']))
        return checkpoint, start_epoch, best_prec1, model_state_dict, optimizer_state_dict
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))
        return None
