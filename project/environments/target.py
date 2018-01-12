import torch

class Targets(object):

    """Structure for holding targets dataset for different processes"""
    def __init__(self, args, dset=None):
        self.Data = []
        self.n = 0

    def load(self, path):
        dset = torch.load(path)
        self.Data.append(dset)
        print('added new dataset')

    def __str__(self):
        return 'Target:\n\nn: {}\nframe: {}\n'.format(self.n, self.frames)

    def __len__(self):
        return self.n

    def check_and_reset(self, masks):
        pass

    def __call__(self):
        return [d[self.n] for d in self.Data]


if __name__ == '__main__':
    from project.utils.arguments import get_args
    args = get_args()

    CurrentTargets = Targets(args)
    CurrentTargets.load(args.target_path)
    CurrentTargets.load(args.target_path2)

    targets = CurrentTargets()

    for t in targets:
        print('type:', type(t))
        print('t:', len(t))

        print(type(t[0]))
        print(t[0].shape)
        print(type(t[1]))
        print(t[1].shape)

