'''
A CLSTMCell used in a CLSTM structure
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class CLSTMCell(nn.Module):
    '''A ConvolutionalLSTM-Cell
    Arguments:

    :param input_shape  - torch.Size
    :param num_features - Number of feature maps in layer
    :param kernel_size  - symmetrical size of feature map kernels
    '''
    def __init__(self, input_shape, num_features, kernel_size=5):
        super(CLSTMCell,self).__init__()
        self.num_features = num_features
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.width = int(input_shape[1])
        self.height = int(input_shape[2])

        # Conv
        self.kernel_size = kernel_size
        self.padding=int((kernel_size-1)/2)
        self.conv = nn.Conv2d(self.input_channels + self.num_features,
                out_channels=4*num_features,
                kernel_size=kernel_size,
                stride=1,
                padding=self.padding,
                bias=True)

        self.output_shape = (None, self.num_features, self.width, self.height)

    def forward(self, x, hidden):
        hx, cx = hidden
        hx, cx = Variable(hx.data, requires_grad=False), Variable(cx.data, requires_grad=False)
        state = torch.cat((x,hx),dim=1) # Concatenate color channels

        i_j_f_o = self.conv(state)
        i, j, f, o = torch.split(i_j_f_o, self.num_features, dim=1)

        c = cx * F.sigmoid(f) + F.sigmoid(i)*F.tanh(j)
        h = F.tanh(c)*F.sigmoid(o)
        return h, (h, c)

    def init_state(self, batch_size, CUDA=False):
        hx = Variable(torch.zeros(batch_size, self.num_features, self.width, self.height))
        cx = Variable(torch.zeros(batch_size, self.num_features, self.width, self.height))
        if CUDA:
            hx, cx = hx.cuda(), cx.cuda()
        return (hx, cx)

    def get_output_shape(self):
        return self.output_shape

    def get_grads(self):
        grads = []
        for p in self.parameters():
            grads.append(p.grad)
        return grads
