import sys
import os
import torch
from tqdm import tqdm

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_video(vid, filename='/tmp/video'):
    fig = plt.figure()
    ims = []
    for frame in tqdm(vid):
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                    repeat_delay=1000)
    name = filename+'.mp4'
    ani.save(name)
    print('Saved video to:', name)

def convert_dir_video(dirpath):
    assert os.path.exists(dirpath)
    print('Converting .pt files to mp4...')
    for root, dirs, files in os.walk(dirpath):
        print('files', files)
        for f in files:
            if f.endswith('.pt'):
                fpath = os.path.join(dirpath, f)
                vid = torch.load(fpath)
                name = fpath[:-3]  # omit extension
                make_video(vid, name)
                print('Removing source file...')
                os.remove(fpath)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        filename = sys.argv[1]
        video_name = sys.argv[2]
        vid = torch.load(filename)
        make_video(vid, video_name)
    else:
        convert_dir_video(sys.argv[1])

