import torch
# import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from project.dynamics_model.utils import Conv2d_out_shape, ConvTranspose2d_out_shape
from project.dynamics_model.CLSTMCell import CLSTMCell


def total_params(in_):
    t = 1
    for x in in_:
        t *= x
    return int(t)


# Networks
class Emb2State(nn.Module):
    '''from embedded to state
    Should be able to return the state even with no prior information.
    damn movement...
    '''
    def __init__(self, in_shape=(64, 7, 7), out_shape=11):
        super(Emb2State, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.conv1 = nn.Conv2d(self.in_shape[0], 16, kernel_size=3, stride=2)
        self.out_shape1 = Conv2d_out_shape(self.conv1, in_shape)

        fc_in = total_params(self.out_shape1)
        self.fc_state = nn.Linear(fc_in, out_shape)

    def forward(self, in_):
        x = F.relu(self.conv1(in_))
        x = x.view(x.size(0), -1)
        return self.fc_state(x)


class StateAction2Emb(nn.Module):
    '''State and action to embedding size
    Arguments:
        state_size - int
        action_sie - int
        emb_shape - (int, int) / torch.Size

    Forward call:
        state - Variable
        action - Variable
    '''
    def __init__(self, state_size, action_size, emb_size, Cuda=False):
        super(StateAction2Emb, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.emb_width = emb_size[0]
        self.emb_height = emb_size[1]
        self.in_size = state_size + action_size
        self.out_size = total_params(emb_size)

        self.fc = nn.Linear(self.in_size, self.out_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1).float()
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 1, self.emb_width, self.emb_height)
        return x


class C_layer(nn.Module):
    '''o'''
    def __init__(self, in_shape, in_channels, out_channels, kernel_size, stride):
        super(C_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_shape = Conv2d_out_shape(self.conv, in_shape)  # output shape of layer
        self.clstm = CLSTMCell(self.out_shape, out_channels, kernel_size=kernel_size)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x, (h1, c1) = self.clstm1(x, self.hidden_states[0])
        x = self.norm1(x)
        self.hidden_states[0] = (h1, c1)  # update hidden state


class Model(nn.Module):
    '''A hardcoded small prediction model to test if the CLSTMCell works.'''
    def __init__(self, input_shape, action_size=2, state_size=11, out_channels=3, feature_list=[32, 64, 64], batch_first=True, CUDA=False):
        super(Model, self).__init__()
        self.CUDA = CUDA
        self.batch_first = batch_first
        self.clstm_list = []
        self.hidden_states = None  # initialized as None, becomes a list after forward call

        self.feature_list = feature_list
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.out_channels = out_channels

        self.action_size = action_size
        self.state_size = state_size

        # Layer 1
        nfeats1 = feature_list[0]
        self.conv1 = nn.Conv2d(self.in_channels, nfeats1, kernel_size=5, stride=2)
        self.out_shape1 = Conv2d_out_shape(self.conv1, input_shape)  # output shape of layer
        self.clstm1 = CLSTMCell(self.out_shape1, nfeats1, kernel_size=5)
        self.clstm_list.append(self.clstm1)
        self.norm1 = nn.InstanceNorm2d(nfeats1)

        # Layer 2
        nfeats2 = feature_list[1]
        self.conv2 = nn.Conv2d(nfeats1, nfeats2, kernel_size=5, stride=2)
        self.out_shape2 = Conv2d_out_shape(self.conv2, self.out_shape1)  # output shape of layer
        self.clstm2 = CLSTMCell(self.out_shape2, nfeats2, kernel_size=5)
        self.clstm_list.append(self.clstm2)
        self.norm2 = nn.InstanceNorm2d(nfeats2)

        # Embedded feature space
        # Assume in_shape=64,7,7 for now

        self.emb2state = Emb2State((64, 7, 7), self.state_size)
        self.state_action_2_emb = StateAction2Emb(self.state_size,
                                                  self.action_size, (7, 7))
        # self.emb2state = nn.Linear()
        # self.state_out = nn.Linear()

        # Layer 3
        nfeats3 = feature_list[2]
        self.conv3_trans = nn.ConvTranspose2d(nfeats2 + 1, nfeats3, kernel_size=5, stride=2, output_padding=1)
        self.out_shape3 = ConvTranspose2d_out_shape(self.conv3_trans, self.out_shape2)  # output shape of layer
        self.clstm3 = CLSTMCell(self.out_shape3, nfeats3, kernel_size=5)
        self.clstm_list.append(self.clstm3)
        self.norm3 = nn.InstanceNorm2d(nfeats3)

        # Layer 4
        nfeats4 = self.out_channels
        self.conv4_trans = nn.ConvTranspose2d(nfeats3, nfeats4, kernel_size=5, stride=2, output_padding=1)
        self.out_shape4 = ConvTranspose2d_out_shape(self.conv4_trans, self.out_shape3)  # output shape of layer
        self.clstm4 = CLSTMCell(self.out_shape4, nfeats4, kernel_size=5)
        self.clstm_list.append(self.clstm4)
        self.norm4 = nn.InstanceNorm2d(nfeats4)

    def forward(self, input, action, state, hidden=None):
        if self.hidden_states is None:
            self.hidden_states = self.init_states(input.size(0))

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input.permute(1, 0, 2, 3, 4)

        seq_len = input.size(1)  # length of sequence
        if len(input) == 4:
            # assume (b,c,h,w)
            seq_len = 1

        state_out = []
        # Sequence
        for t in range(seq_len):
            x = self.conv1(input[:, t])  # the t:th sequence datum
            x, (h1, c1) = self.clstm1(x, self.hidden_states[0])
            x = self.norm1(x)
            self.hidden_states[0] = (h1, c1)  # update hidden state

            x = self.conv2(x)
            x, (h2, c2) = self.clstm2(x, self.hidden_states[1])
            emb = self.norm2(x)
            self.hidden_states[1] = (h2, c2)  # update hidden state

            # Embedding done. Incooperate state/action
            st_out = self.emb2state(emb)
            state_out.append(st_out.unsqueeze(1))
            y = self.state_action_2_emb(state[:, t], action[:, t])
            emb = torch.cat((emb, y), dim=1)

            # Reconstruction
            x = self.conv3_trans(emb)
            x, (h3, c3) = self.clstm3(x, self.hidden_states[2])
            x = self.norm3(x)
            self.hidden_states[2] = (h3, c3)  # update hidden state

            x = self.conv4_trans(x)
            x, (h4, c4) = self.clstm4(x, self.hidden_states[3])
            x = self.norm4(x)
            self.hidden_states[3] = (h4, c4)  # update hidden state

        # Should the rgb-output of the network be squeezed through sigmoid ?
        # Common normalization range of images.
        state_out = torch.cat(state_out, dim=1)
        return x, state_out, self.hidden_states

    def init_states(self, batch_size):
        states = []
        for i in range(len(self.clstm_list)):
            states.append(self.clstm_list[i].init_state(batch_size, CUDA=self.CUDA))
        return states

    def reset_hidden(self):
        self.hidden_states = None

    def cuda(self, **args):
        super(Model, self).cuda(**args)
        self.CUDA = True

    def cpu(self, **args):
        super(Model, self).cpu(**args)

    def predict_future(self, model, rgb, state, action, next_actions, step=5, imshow=True):
        future = {'rgb': [], 'state': []}
        for i in range(step):
            x, s, _ = model(rgb, action=action, state=state)
            future['rgb'].append(x.data)
            future['state'].append(s.data)
            x = x.unsqueeze(1)
            s = s[:, -1].unsqueeze(1)
            # Update inputs (should be a better way)
            rgb_ = Variable(rgb.data[:, 1:])
            state_ = Variable(state.data[:, 1:])
            action_ = Variable(action.data[:, 1:])
            rgb = torch.cat((rgb_, x), dim=1)
            state = torch.cat((state_, s), dim=1)
            action = torch.cat((action_, next_actions[i]), dim=1)
        return future


