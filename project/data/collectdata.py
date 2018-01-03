def DataGenerator(dpoints=1000, prob=0.03):
    """ DataGenerator runs some episodes and randomly saves rgb, state pairs
    Kwargs:
        :dpoints : Number of data points to collect
        :prob    : probability of chosing a state/obs pair

    Returns:
        dict
    """
    from numpy.random import uniform
    args.RGB = True  # to be safe
    env = TargetHumanoid(args)
    s, obs = env.reset()
    t = 0
    states, obs_list = [], []
    while len(states) < dpoints:
        s, obs, _, d, _ = env.step(env.action_space.sample())
        t += 1
        if uniform() < prob:
            states.append(s)
            obs_list.append(obs)
        if d:
            s=env.reset()
            t=0
    return {'states': states, 'obs':obs_list}

def save_data(dpoints):
    import torch
    data = DataGenerator(dpoints)
    name = '/home/erik/DATA/humanoid/test.pt'
    torch.save(data, name)

def show_obs_state(datadict):
    """Prints out state and previews corresponding observation
    Args:
        datadict : dict containing states and obs
    """
    # import cv2
    import matplotlib.pyplot as plt
    for s, obs in zip(datadict['states'], datadict['obs']):
        print('State: ', s)
        # cv2.imshow('', obs)
        plt.imshow(obs)
        plt.pause(0.1)
        input('Enter when done')

