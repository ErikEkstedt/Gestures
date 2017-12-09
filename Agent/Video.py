import torch
import gym
from itertools import count
from memory import StackedState

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plt_watch(rgb_list):
    for f in rgb_list:
        plt.imshow(f)
        plt.pause(0.01)

def cv2_watch(rgb_list):
    import cv2
    for i,f in enumerate(rgb_list):
        cv2.imshow(str(i), f)
        if cv2.waitKey(self.wait) & 0xFF == ord('q'):
            break

def make_video(vid, name):
    fig = plt.figure()
    ims = []
    for frame in vid:
        im = plt.imshow(frame, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                    repeat_delay=1000)
    ani.save(name+'.mp4')

def is_video_same(vid):
    for i in range(len(vid)-1):
        print(v[i] == v[i+1])


if __name__ == '__main__':
    import sys
    vid = torch.load(sys.argv[1])
    # cv2_watch(vid)
    make_video(vid, 'Did it')
    # plt_watch(vid)
