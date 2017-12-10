import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='PPOAgent')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')

    # gym
    parser.add_argument('--num-processes', type=int, default=4,
                       help='Number of processors used (default: 4)')
    parser.add_argument('--env-id', default='CustomReacher',
                       help='Environment used (default: CustomReacher)')

    # PPO Loss
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='policy learning rate (default: 4e-4)')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='epsilon value(default: 1e-8)')
    parser.add_argument('--no-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--max-grad-norm', type=float, default=5,
                        help='ppo clip parameter (default: 5)')


    # PPO Training
    parser.add_argument('--num-frames', type=int, default=int(10e6),
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--num-steps', type=int, default=2048,
                        help='number of exploration steps in ppo (default: ?)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='ppo batch size (default: 256)')
    parser.add_argument('--max-episode-length', type=int, default=100000,
                        help='maximum steps in one episode (default: 10000)')
    parser.add_argument('--ppo-epoch', type=int, default=8,
                        help='number of ppo epochs, K in paper (default: 8)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden neurons in policy (default: 128)')
    parser.add_argument('--std-start', type=float, default=-0.6,
                        help='std-start (Hyperparams for Roboschool in paper)')
    parser.add_argument('--std-stop', type=float, default=-1.7,
                        help='std stop (Hyperparams for Roboschool in paper)')

    # Test
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--test-interval', type=int,  default=50,
                        help='how many updates/test (default: 50)')
    parser.add_argument('--num-test', type=int, default=10,
                        help='Number of test after training (default: 100)')

    # Log
    parser.add_argument('--vis-interval', type=int, default=1,
                        help='vis interval, one log per n updates (default: 1)')
    parser.add_argument('--log-dir', default='/tmp/',
                        help='directory to save agent logs (default: /tmp/)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval in console, one log per n updates (default: 1)')

    # Boolean
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    return args


class FakeArgs:
    def __init__(self):
        self.seed               = 1
        self.num_processes      = 2
        self.pi_lr              = 3e-4
        self.eps                = 1e-5
        self.use_gae            = True
        self.gamma              = 0.99
        self.tau                = 0.95
        self.entropy_coef       = 0.01
        self.clip_param         = 0.2
        self.max_grad_norm      = 2

        # PPO Training
        self.num_frames         = int(1e6)
        self.num_steps          = 2048
        self.batch_size         = 64
        self.max_episode_length = 10000
        self.ppo_epoch          = 8
        self.num_stack          = 1
        self.hidden             = 64
        # self.fixed_std          = False
        # self.std                = 0.2

        # Test
        self.no_test            = False
        self.test_interval      = 1
        self.max_test_length    = 10000
        self.num_test           = 10

        # Log
        self.vis_interval       = 1
        self.log_dir            = '/tmp/'
        self.log_interval       = '/tmp/'
        self.save_interval      = '/tmp/'
        self.save_dir           = '/tmp/'

        # Boolean
        self.no_cuda            = True
        self.no_vis             = True
        self.test_render        = True

        self.cuda = not self.no_cuda and torch.cuda.is_available()
        self.vis = not self.no_vis

