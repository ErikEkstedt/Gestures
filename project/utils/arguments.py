import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description='PPOAgent')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--dpoints', type=int, default=500000)
    parser.add_argument('--episodes', type=int, help='complete episode trajectory to gather for mimic', default=10)
    parser.add_argument('--update-target', type=int, default=10, help='Number of frames between target update (default: 10)')

    # === Environment ===
    parser.add_argument('--env-id', default='Social')
    parser.add_argument('--dof', type=int, default=2)
    parser.add_argument('--video-w', type=int, default=40)
    parser.add_argument('--video-h', type=int, default=40)
    parser.add_argument('--video-c', type=int, default=3)
    parser.add_argument('--MAX_TIME', type=int, default=300)
    parser.add_argument('--gravity', type=float, default=9.81)
    parser.add_argument('--power', type=float, default=0.5)
    parser.add_argument('--RGB', action='store_true', default=False)
    parser.add_argument('--COMBI', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False)

    # === Reward =====
    parser.add_argument('--potential-constant',   type=float, default=100)
    parser.add_argument('--electricity-cost',     type=float, default=0.1)
    parser.add_argument('--stall-torque-cost',    type=float, default=0.01)
    parser.add_argument('--joints-at-limit-cost', type=float, default=0.01)
    parser.add_argument('--r1', type=float, default=1.0)
    parser.add_argument('--r2', type=float, default=1.0)

    # === PPO Loss ===
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon value(default: 1e-8)')
    parser.add_argument('--no-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.0, help='entropy term coefficient (default: 0.0)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--max-grad-norm', type=float, default=5, help='ppo clip parameter (default: 5)')

    # === PPO Training ===
    parser.add_argument('--continue-training', action='store_true', default=False)
    parser.add_argument('--num-frames', type=int, default=int(3e6), help='number of frames to train (default: 3e6)')
    parser.add_argument('--num-steps', type=int, default=2048, help='number of exploration steps in ppo (default: ?)')
    parser.add_argument('--batch-size', type=int, default=256, help='ppo batch size (default: 256)')
    parser.add_argument('--max-episode-length', type=int, default=1000, help='maximum steps in one episode (default: 1000)')
    parser.add_argument('--ppo-epoch', type=int, default=8, help='number of ppo epochs, K in paper (default: 8)')
    parser.add_argument('--num-stack', type=int, default=1, help='number of frames to stack (default: 1)')
    parser.add_argument('--std-start', type=float, default=-0.6, help='std-start (Hyperparams for Roboschool in paper)')
    parser.add_argument('--std-stop', type=float, default=-1.7, help='std stop (Hyperparams for Roboschool in paper)')
    parser.add_argument('--seed', type=int, default=99, help='random seed (default: 99)')

    parser.add_argument('--pi-lr', type=float, default=3e-4, help='policy learning rate (default: 3e-5)')
    parser.add_argument('--adjust-lr', action='store_true', default=False)
    parser.add_argument('--adjust-lr-interval', type=int, default=20, help='Updates (default: 20)')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='decay factor for learning rate (default: 0.9)')
    parser.add_argument('--pi-end-lr', type=float, default=3e-5, help='policy learning rate (default: 3e-5)')

    # === MODEL ===
    parser.add_argument('--feature-maps', nargs='+', type=int, default=[64,64,8])
    parser.add_argument('--kernel-sizes', nargs='+', type=int, default=[5,5,5])
    parser.add_argument('--strides', nargs='+', type=int, default=[2,2,2])

    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden neurons in policy (default: 128)')
    parser.add_argument('--use_target_state', action='store_true', default=False)

    parser.add_argument('--epochs', type=int, default=200, help='Epochs used for understanding training(default: 128)')
    parser.add_argument('--cnn-lr', type=float, default=3e-4, help='cnn learning rate (default: 3e-4)')
    parser.add_argument('--save-interval', type=float, default=10, help='Save interval (default: 10)')

    # === TEST ===
    parser.add_argument('--no-test', action='store_true', default=False, help='disables test during training')
    parser.add_argument('--test-interval', type=int,  default=50, help='how many updates/test (default: 50)')
    parser.add_argument('--num-test', type=int, default=5, help='Number of test episodes during test (default: 20)')
    parser.add_argument('--test-thresh', type=int, default=1000000, help='number of frames before test (default: 1000000)')


    sdpath = os.path.join(os.path.dirname(__file__), "../results/BestDictCombi4710400_65.577.pt")
    parser.add_argument('--state-dict-path', default=sdpath, help='Path to state_dict to load')
    parser.add_argument('--scale', type=int, default=1, help='scale image for enjoy.py')

    targetpath = os.path.join(os.path.dirname(__file__), "../results/socialtargets_s4_o40-40-3_n5000_1.pt")
    parser.add_argument('--target-path', default=targetpath, help='Path to target to load')
    parser.add_argument('--target-path2', default=targetpath, help='Path to target to load')
    parser.add_argument('--target-path3', default=targetpath, help='Path to target to load')
    parser.add_argument('--random-targets', action='store_true', default=False)

    # === LOG ===
    parser.add_argument('--vis-interval', type=int, default=1, help='vis interval, one log per n updates (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, help='log interval in console, one log per n updates (default: 1)')
    parser.add_argument('--log-dir', default='/tmp', help='directory to save agent logs')
    parser.add_argument('--filepath', default='/tmp/file_created_by_project_args', help='Filepath')


    # === Boolean ===
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-vis', action='store_true', default=False, help='disables visdom visualization')
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    return args
