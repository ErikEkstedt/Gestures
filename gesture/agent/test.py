from itertools import count
import os
import numpy as np
from tqdm import tqdm, trange
from gesture.agent.memory import Current
from gesture.utils.utils import record, get_model


def Test_and_Save_Video(env, targets, state_dict, args, frames, Model=None):
    '''
    Test with video
    :param env   - Reacher/HUmanoid environment
    :param targets    - Target class
    :param Model      - The policy network
    :param state_dict - nn.Module.state_dict
    :param verbose    - Boolean, be verbose
    '''
    if args.record:
        import skvideo.io
        name = "{}-test_frame{}.mp4".format(args.env_id, frames)
        name = os.path.join(args.result_dir, name)
        writer = skvideo.io.FFmpegWriter(name)

    # === Target dims ===
    st_sample, ob_sample = targets()  #random index
    ot_shape = ob_sample.shape
    st_shape = st_sample.shape[0]

    # == Model
    s_shape = env.state_space.shape[0]    # Joints state
    o_shape = env.observation_space.shape # RGB
    ac_shape = env.action_space.shape[0]   # Actions
    current = Current(num_processes=1,
                      num_stack=args.num_stack,
                      state_dims=s_shape,
                      starget_dims=st_shape,
                      obs_dims=o_shape,
                      otarget_dims=ot_shape,
                      ac_shape=ac_shape)

    if Model is None:
        pi, _ = get_model(current, args)
    else:
        pi = Model(s_shape=current.s_shape,
                   st_shape=current.st_shape,
                   o_shape=current.o_shape,
                   ot_shape=current.ot_shape,
                   a_shape=current.ac_shape,
                   feature_maps=args.feature_maps,
                   kernel_sizes=args.kernel_sizes,
                   strides=args.strides,
                   args=args)

    pi.load_state_dict(state_dict)
    pi.eval()

    total_reward = 0
    for i in trange(args.num_test):
        env.set_target(targets())
        state, s_target, obs, o_target = env.reset()
        for j in count(1):
            current.update(state, s_target, obs, o_target)

            s, st, o, ot = current()
            value, action = pi.act(s, st, o, ot)
            cpu_actions = action.data.cpu().numpy()[0]

            # Observe reward and next state
            state, s_target, obs, o_target, reward, done, info = env.step(cpu_actions)

            if args.record and j % 2 == 0: # every other frame
                record(env, writer)

            # If done then update final rewards and reset episode reward
            total_reward += reward
            if done:
                break

    if args.record:
        writer.close()
    return total_reward/args.num_test
