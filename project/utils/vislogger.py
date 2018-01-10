from visdom import Visdom
import numpy as np
import torch

# Run 'python -m visdom.server'

def to_numpy(x):
    if type(x) is torch.Tensor:
        x = x.cpu().numpy()
    elif type(x) is torch.autograd.Variable:
        x = x.data.cpu().numpy()
    elif type(x) is float or type(x) is int:
        x = np.array([x])
    else:
        x = np.array(x)
    return x


# plot errors/ mean+std
# https://github.com/facebookresearch/visdom/issues/201
def make_errors(ys, xs=None, color=None, name=None):
    if xs is None:
        xs = [list(range(len(y))) for y in ys]

    minX = min([len(x) for x in xs])

    xs = [x[:minX] for x in xs]
    ys = [y[:minX] for y in ys]

    assert all([(len(y) == len(ys[0])) for y in ys]), \
        'Y should be the same size for all traces'

    assert all([(x == xs[0]) for x in xs]), \
        'X should be the same for all traces'

    y = np.array(ys)
    yavg = np.mean(y, 0)
    ystd = np.std(y, 0)

    err_traces = [
        dict(x=xs[0], y=yavg.tolist(), mode='lines', name=name,
            line=dict(color=color)),
        dict(
            x=xs[0] + xs[0][::-1],
            y=(yavg + ystd).tolist() + (yavg - ystd).tolist()[::-1],
            fill='tozerox',
            fillcolor=(color[:-1] + str(', 0.2)')).replace('rgb', 'rgba')
                        if color is not None else None,
            line=dict(color='transparent'),
            name=name + str('_error') if name is not None else None,
        )
    ]
    return err_traces, xs, ys


class VisLogger(object):
    def __init__(self, args, desc=True):
        ''' Visdom logger
        :param description_list     list contining strings
        :param log_dir              string, directory to log in
        :param name                 string, (optional) specific name for log subfolder
        '''
        self.viz = Visdom()
        assert self.viz.check_connection(), "Server not found.\n \
            Make sure to execute 'python -m visdom.server' \
            (with the right python version) "

        self.args_string = self.args_to_list(args)

        # Log args in vis-text and print to console
        if desc:
            self.description = self.viz.text('')
            for line in self.args_string:
                self.viz.text(line, win=self.description, append=True)

        self.windows = {}

    def print_console(self):
        for s in self.args_string:
            print(s)

    def args_to_list(self, args):
        l = []
        for arg, value in args._get_kwargs():
            s = "{}: {}".format(arg, value)
            l.append(s)
        return l

    def line_update(self, Xdata, Ydata, name):
        '''
        :param Xdata - torch.Tensor or float
        :param Ydata - torch.Tensor or float
        :param name - string
        '''
        Xdata = to_numpy(Xdata)
        Ydata = to_numpy(Ydata)

        if name in self.windows.keys():
            self.viz.line(Y=Ydata, X=Xdata, win=self.windows[name], update='append')
        else:
            self.windows[name] = self.viz.line(Y=Ydata, X=Xdata,
                    opts=dict(showlegend=True, xlabel='Frames',
                    ylabel=name, title=name,),)

    def scatter_update(self, Xdata, Ydata, name):
        '''
        :param Xdata - torch.Tensor or float
        :param Ydata - torch.Tensor or float
        :param name - string
        '''
        Xdata = to_numpy(Xdata)
        Ydata = to_numpy(Ydata)
        X = np.stack((Xdata, Ydata)).T  # Nx2 shape
        if name in self.windows.keys():
            self.viz.scatter(X, win=self.windows[name], update='append')
        else:
            self.windows[name] = self.viz.scatter(X, opts=dict(showlegend=True,
                                                            xlabel='Frames',
                                                            ylabel=name,
                                                            title=name,),)

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


def test_line_scatter(logger):
    x = [1,2,3,4,5,6,7]
    y = [1,2,3,4,5,6,7]
    logger.scatter_update(x,y, 'Scatter')
    logger.line_update(x,y, 'Line')
    x = [8]
    y = [4]
    logger.scatter_update(x,y, 'Scatter')
    logger.line_update(x,y, 'Line')

def test_box(logger):
    box = torch.Tensor([1,2,3,5,9,5,3,2,1])
    logger.bar_update(box, name='box1')
    time.sleep(1)
    logger.bar_update(box, name='box1')
    time.sleep(1)
    logger.bar_update(3*box, name='box1')
    time.sleep(1)
    logger.bar_update(3*box, name='box2')


if __name__ == '__main__':
    from project.utils.arguments import get_args
    import torch
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    args = get_args()
    logger = VisLogger(args, False)

    y = [(1,2,3), (1,2,3), (1,2,3)]
    err_traces, xs, ys = make_errors(y)
    print( 'err_traces', err_traces)
    print( 'xs', xs)
    print( 'ys', ys)

