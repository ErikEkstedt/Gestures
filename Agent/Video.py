import torch
import gym
from itertools import count
from memory import StackedState

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VideoPlayer(object):
    def __init__(self, Video, fps):
        import cv2
        self.video = Video
        self.fps = fps
        self.wait = 1/fps

    def __call__(self, use_cv2=False):
        print('Playing video')
        import cv2
        if use_cv2:
            for i, frame in enumerate(self.video):
                # print(frame)
                # Display the resulting frame
                cv2.imshow(str(i), frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(self.wait) & 0xFF == ord('q'):
                    break
        else:
            for i, frame in enumerate(self.video):
                # Display the resulting frame
                # print(frame.shape)
                # print(frame.dtype)
                plt.imshow(frame)
                # plt.pause(self.wait)
                plt.pause(0.11)

def make_video(vid, name):
    fig = plt.figure()
    ims = []
    for frame in vid:
        im = plt.imshow(frame, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
                                    repeat_delay=1000)
    ani.save(name+'.mp4')

def watch_video():
    import sys
    print(sys.argv[1])
    vid = torch.load(sys.argv[1])
    vp = VideoPlayer(vid, 25)
    vp()


if __name__ == '__main__':
    watch_video()

