import sys
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


if __name__ == '__main__':
    filename = sys.argv[1]
    video_name = sys.argv[2]
    vid = torch.load(filename)
    make_video(vid, video_name)
