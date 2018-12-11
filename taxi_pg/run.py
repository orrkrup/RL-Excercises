#!/usr/bin/env python

import argparse
import torch
import gym
from tqdm import trange

from taxi_pg import PGModel, to_categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_state(obs_dim, inds):
    v = to_categorical(inds, obs_dim)
    return v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--modelfilename', type=str, default='./taxi_pg/opt_model', help='Filename for input model')
    parser.add_argument('-s', '--save_video', action='store_true', help='Save video of model running')
    parser.add_argument('-e', '--episodes', type=int, default=100, help='How many episodes to run model for')

    args = parser.parse_args()

    env = gym.make('Taxi-v2')
    obs_dim = env.observation_space.n
    model = PGModel(obs_dim, 50, env.action_space.n).to(device)
    model.load_state_dict(torch.load(args.modelfilename))
    model.eval()

    eps = 0.01
    episodes = args.episodes
    cr = 0
    for ep in trange(episodes):
        vobs = env.reset()
        obs = make_state(obs_dim, torch.LongTensor([vobs]))
        done = False
        while not done:
            p, _ = model(obs)
            act = torch.multinomial(torch.nn.functional.softmax(p, dim=-1), 1).item()

            vobs, r, done, _ = env.step(act)
            obs = make_state(obs_dim, torch.LongTensor([vobs]))

            cr += r

    avg_r = cr * 1.0 / episodes

    print("Model Evaluation done. Average reward: {}".format(avg_r))

