import sys
import os
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def BGR_to_RGB(im):
    r, g, b = im
    if type(im) is np.ndarray:
        return np.stack((b,g,r), axis=0)
    else:
        return torch.stack((b,g,r), dim=0)

def BGR_to_RGB(im):
    b, g, r = im
    if type(im) is np.ndarray:
        return np.stack((r, g,b), axis=0)
    else:
        return torch.stack((r, g,b), axis=0)

def make_video(vids, targets, filename='/tmp/video'):
    fig = plt.figure()
    ims = []
    t = 0
    for i, frame in enumerate(tqdm(vids)):
        if i % 150 == 0:
            target_im = targets[t][0]
            t += 1
        frame = torch.from_numpy(frame.transpose((2,0,1))).float()
        frame = BGR_to_RGB(frame)
        imglist = [frame, target_im]
        img = make_grid(imglist, padding=5).numpy()
        img *= 255
        img = img.astype('uint8').transpose((1,2,0))
        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
    name = filename+'.mp4'
    ani.save(name)
    print('Saved video to:', name)

def combine_video_dir(dirpath):
    assert os.path.exists(dirpath)
    print('Converting .pt files to mp4...')
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            if f.endswith('.pt'):
                fpath = os.path.join(dirpath, f)
                vids, targets = torch.load(fpath)
                name = fpath[:-3]  # omit extension
                print('Converting', name)
                make_video(vids, targets, name)

def delete_sources(dirpath):
    assert os.path.exists(dirpath)
    print('Removing source files...')
    for root, dirs, files in os.walk(dirpath):
        for f in tqdm(files):
            if f.endswith('.pt'):
                fpath = os.path.join(dirpath, f)
                os.remove(fpath)
    print('Directory is clean!')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        video_name = sys.argv[2]
        vid, targets = torch.load(filename)
        make_video(vid, targets, video_name)
    else:
        dirpath = sys.argv[1]
        combine_video_dir(dirpath)
        ans = input('Wish to delete source files? (yes/no)\n>')
        if ans == "yes":
            delete_sources(dirpath)

