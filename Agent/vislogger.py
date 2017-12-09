from visdom import Visdom
import datetime
import os
import numpy as np
import torch

# Run 'python -m visdom.server'
# First goal - make visualizer that draws reward vs frames

def get_today():
    t = datetime.date.today().ctime().split()[1:3]
    s = "".join(t)
    return s

def to_numpy(x):
    if type(x) is torch.Tensor:
        x = x.cpu().numpy()
    elif type(x) is torch.autograd.Variable:
        x = x.data.cpu().numpy()
    elif type(x) is float or int:
        x = np.array([x])
    else:
        # assume x is numpy
        pass
    return x

class VisLogger(object):
    def __init__(self, description_list=None, log_dir='/tmp/', name=None):
        ''' Visdom logger
        :param description_list     list contining strings
        :param log_dir              string, directory to log in
        :param name                 string, (optional) specific name for log subfolder
        '''
        self.viz = Visdom()
        assert self.viz.check_connection(), "Server not found.\n \
            Make sure to execute 'python -m visdom.server' \
            (with the right python version) "

        self._mkdirs(log_dir, name)  # Creates: self.log_dir, self.checkpoint_dir
        self.best_score = None       # best score achieved.

        # Todo add args and agent properties in description box
        self.description = self.viz.text('')
        for line in description_list:
            self.viz.text(line, win=self.description, append=True)
        self.windows = {}

    def get_logdir(self):
        return self.log_dir, self.video_dir, self.checkpoint_dir

    def _mkdirs(self, log_dir, name):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        day = get_today()
        if name:
            log_dir = os.path.join(log_dir, name, day)
        else:
            log_dir = os.path.join(log_dir, day)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        run = 0
        while os.path.exists("%s/run-%d" % (log_dir, run)):
            run += 1

        self.log_dir = "%s/run-%d" % (log_dir, run)
        self.video_dir = "%s/video" % (self.log_dir)
        self.checkpoint_dir = "%s/checkpoints" % (self.log_dir)
        os.mkdir(self.log_dir)
        os.mkdir(self.checkpoint_dir)

    def line_update(self, Xdata, Ydata, name):
        '''
        :param Xdata - torch.Tensor or float
        :param Ydata - torch.Tensor or float
        :param name - string
        '''
        Xdata = to_numpy(Xdata)
        Ydata = to_numpy(Ydata)

        if name in self.windows.keys():
            self.viz.updateTrace(Y=Ydata, X=Xdata, win=self.windows[name], append=True)
        else:
            self.windows[name] = self.viz.line(Y=Ydata, X=Xdata,
                    opts=dict(showlegend=True, xlabel='Frames',
                    ylabel=name, title=name,),)

    def bar_update(self, X, name):
        '''
        :param X - torch.Tensor or float
        :param name - string
        '''
        X = to_numpy(X)
        X /= X.max()
        if name in self.windows.keys():
            # update existing
            self.viz.close(self.windows[name])
            self.windows[name] = self.viz.bar(X=X,
                    opts=dict(showlegend=True, title=name,),)
        else:
            self.windows[name] = self.viz.bar(X=X,
                    opts=dict(showlegend=True, title=name,),)

    def save(self):
        print('Saving the visdom server content (~/.visdom)')
        self.viz.save([self.viz.env])

def test():
    import torch
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    logger = VisLogger()
    box = torch.Tensor([1,2,3,5,9,5,3,2,1])
    logger.bar_update(box, name='box1')
    time.sleep(1)
    logger.bar_update(box, name='box1')
    time.sleep(1)
    logger.bar_update(3*box, name='box1')
    time.sleep(1)
    logger.bar_update(3*box, name='box2')


if __name__ == '__main__':
    test()
