from __future__ import print_function

from mss.linux import MSS as mss
from subprocess import Popen, PIPE
import numpy as np


class ObsRGB(object):
    ''' A class to retrieve rgb data from Robot view in choregraphe
    (Robot view has to be detached from choregraph [double click on top of Robot view]) '''

    def __init__(self, n=60):
        x, y, w, h, l = self.get_robot_window()
        self.l = l
        self.n = 60
        self.monitor = {'top': y+n, 'left': x, 'width': w, 'height': h-n}
        self.sct = mss()

    def get_rgb(self):
        sct_img = self.sct.grab(self.monitor)
        return np.array(sct_img)[:,:,:3]

    def get_robot_window(self):
        ''' An ugly way of getting the window info needed'''
        root = Popen(['xwininfo', '-name', 'Robot view'], stdout=PIPE)
        x, y, w, h = None, None, None, None
        l = []
        for line in root.stdout:
            sline = str(line)
            l.append(sline)
            if "Width" in sline:
                s = [s for s in sline.split()]
                w = s[-1]
            elif "Height" in sline:
                s = [s for s in sline.split()]
                h = s[-1]
            elif "Absolute upper-left X" in sline:
                s = [s for s in sline.split()]
                x = s[-1]
            elif "Absolute upper-left Y" in sline:
                s = [s for s in sline.split()]
                y = s[-1]
            else:
                continue
        return int(x), int(y), int(w), int(h), l

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=0)

    args = parser.parse_args()
    obs = ObsRGB(args)
    img = obs.get_rgb()
    print(type(img))
    print(img.dtype)
    plt.imshow(img)
    plt.show()
    print('DOne')


