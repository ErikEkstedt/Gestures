import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='AgentRoboSchool PPO Algorithm')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')


    # gym
    parser.add_argument('--test-render', action='store_true', default=False,
            help='Render during test')
    parser.add_argument('--num-processes', type=int, default=2,
                       help='Number of processors used (default: 2)')

    # PPO Loss
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='policy learning rate (default: 4e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='epsilon value(default: 1e-5)')
    parser.add_argument('--use-gae', action='store_true', default=False,
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
    parser.add_argument('--num-frames', type=int, default=int(1e6),
                        help='number of frames to train (default: 1e6)')
    parser.add_argument('--num-steps', type=int, default=2048,
                        help='number of exploration steps in ppo (default: ?)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='ppo batch size (default: 64)')
    parser.add_argument('--max-episode-length', type=int, default=10000,
                        help='maximum steps in one episode (default: 10000)')
    parser.add_argument('--ppo-epoch', type=int, default=8,
                        help='number of ppo epochs (default: 8)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden neurons in policy (default: 128)')
    parser.add_argument('--fixed-std', action='store_true', default=False,
            help='Use a fixed standard deviation for actions')
    parser.add_argument('--std', type=float, default=0.5,
            help='Value for action standard  deviation (default: 0.5)')

    # Test
    parser.add_argument('--no-test', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--test-interval', type=int,  default=100,
            help='how many upgrades/test (default: 100)')
    parser.add_argument('--max-test-length', type=int, default=200,
                        help='maximum steps in a test episode (default: 700)')
    parser.add_argument('--num-test', type=int, default=5,
            help='Number of test after training (default: 5)')


    # Log
    parser.add_argument('--vis-interval', type=int, default=10,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--log-dir', default='/tmp/',
            help='directory to save agent logs (default: /tmp/)')

    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')

    # Cuda
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
        self.cuda = False
        self.seed = 1
        self.render = True

        self.fixed_std = True
        self.std = 0.5
        self.pi_lr = 3e-4                  # policy learning rate
        self.f_lr = 1e-3                   # f_net learning rate
        self.entropy_coef = 0
        self.use_gae = True
        self.gamma = 0.99
        self.tau = 0.95
        self.eps = 1e-5

        self.total_frames = int(2e6)
        self.ppo_epoch = 10                # K
        self.batch_size = 64
        self.clip_param = 0.3              # eps in clip
        self.num_steps = 2048              # horizon
        self.num_stack = 1                 # imgs/states-observations to stack in one state
        self.num_frames = int(2e6)         # total frames seen in training
        self.max_episode_length = 1000

        self.vis = True                    # use visdom for graphs
        self.vis_interval = 10
        self.log_interval = 100
        self.save_interval = 1000

        try:
            os.mkdir('/tmp/AgentPepper')
        except:
            pass

        self.log_dir = "/tmp/AgentPepper"
        self.save_dir = "/tmp/AgentPepper"
        self.algo = "Agent-Pepper"

        self.IP = "localhost"
        self.PORT = 37901



