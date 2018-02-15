import torch
import datetime
import os
import numpy as np

from torchvision.utils import make_grid
import cv2

# ======================== #
# Render                   #
# ======================== #
def rgb_render(obs, title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return

def rgb_tensor_render(obs, scale=(1,1), title='tensor_obs'):
    assert type(obs) is torch.Tensor
    assert len(obs.shape) == 3
    obs = obs.permute(1,2,0)
    im = obs.numpy().astype('uint8')
    # rgb_render(im, title)
    render_and_scale(im, scale=scale, title=title)

def render_and_scale(obs, scale=(1, 1), title='obs'):
    ''' cv2 as argument such that import is not done redundantly'''
    height, width = obs.shape[:2]
    obs = cv2.resize(obs,(scale[0]*width, scale[0]*height), interpolation = cv2.INTER_CUBIC)
    cv2.imshow(title, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print('Stop')
        return

def get_model(current, args):
    if 'SemiCombine' in args.model:
        from gesture.models.combine import SemiCombinePolicy
        Model = SemiCombinePolicy
        pi = Model(s_shape=current.s_shape,
                   st_shape=current.st_shape,
                   o_shape=current.o_shape,
                   ot_shape=current.ot_shape,
                   a_shape=current.ac_shape,
                   feature_maps=args.feature_maps,
                   kernel_sizes=args.kernel_sizes,
                   strides=args.strides,
                   args=args)
    elif 'Combine' in args.model:
        from gesture.models.combine import CombinePolicy
        Model = CombinePolicy
        pi = Model(s_shape=current.s_shape,
                   st_shape=current.st_shape,
                   o_shape=current.o_shape,
                   ot_shape=current.ot_shape,
                   a_shape=current.ac_shape,
                   feature_maps=args.feature_maps,
                   kernel_sizes=args.kernel_sizes,
                   strides=args.strides,
                   args=args)
    else:
        from gesture.models.modular import MLPPolicy
        Model = MLPPolicy
        in_size = current.st_shape + current.s_shape
        pi = Model(input_size=in_size, a_shape=current.ac_shape, args=args)
    return pi, Model

def get_targets(args):
    from agent.memory import Targets
    from utils.utils import load_dict
    print('\nTraining:', args.train_target_path)
    train_dict = load_dict(args.train_target_path)

    print('\nTesting:', args.test_target_path)
    test_dict = load_dict(args.test_target_path)

    targets = Targets(args.num_proc, datadict=train_dict)
    test_targets = Targets(1, datadict=test_dict)

    if not args.speed:
        targets.remove_speed(args.njoints)  # args.njoints is initialized in "env_to_args"
        test_targets.remove_speed(args.njoints)

    s_target, o_target = targets.random_target()
    s_te, o_te = test_targets.random_target() # check to have same dims as training set
    assert s_target.shape == s_te.shape, 'training and test shapes do not match'
    assert o_target.shape == o_te.shape, 'training and test shapes do not match'
    return targets, test_targets

def log_print(agent, dist_entropy, value_loss, floss, action_loss, j):
    print("\nUpdate: {}, frames:    {} \
          \nAverage final reward:   {}, \
          \nentropy:                {:.4f}, \
          \ncurrent value loss:     {:.4f}, \
          \ncurrent policy loss:    {:.4f}".format(j,
                (j + 1) * agent.args.num_steps * agent.args.num_proc,
                agent.final_rewards.mean(),
                -dist_entropy.data[0],
                value_loss.data[0],
                action_loss.data[0],))


def record(env, writer):
    human, _, target = env.render('all_rgb_array')  # (W, H, C)
    H, W = human.shape[:2]
    # target: (w,h,3) -> (3, W,H)
    target = cv2.resize(target,(W, H), interpolation = cv2.INTER_CUBIC)
    target = target.transpose((2,0,1))
    # human: (W,H, 3) -> (3, W,H)
    human = human.transpose((2,0,1))
    imglist = [torch.from_numpy(human), torch.from_numpy(target)]
    img = make_grid(imglist, padding=5).numpy()
    img = img.transpose((1,2,0))
    writer.writeFrame(img)


def adjust_learning_rate(optimizer, decay=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def adjust_learning_rate2(optimizer, args, frame):
    ratio = frame/args.num_frames
    LR = args.pi_lr -(args.pi_lr - args.pi_end_lr)*ratio
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR


def save_dict(datadict, filename):
    """Save a dict with h5py
    Args:
        datadict   dict: {
        filename   string: full filepath
    """
    with h5py.File(filename, 'w') as hdf:
        for k, v in datadict.items():
            hdf.create_dataset(k, data=v)


def load_dict(filename):
    """Load a dict with h5py
    Args:
        filename   string: full filepath
    """
    datadict = {}
    with h5py.File(filename, 'r') as hdf:
        for k in hdf.keys():
            datadict[k] = list(hdf.get(k))
    return datadict


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

