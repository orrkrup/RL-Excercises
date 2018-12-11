#!/usr/bin/env python

import argparse
import torch
import gym
from .taxi_pg import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--modelfilename', type=str, default='opt_model', help='Filename for input model')
    parser.add_argument('-s', '--save_video', action='store_true', help='Save video of model running')
    parser.add_argument('-e', '--episodes', type=int, default=100, help='How many episodes to run model for')

    args = parser.parse_args()

    env = gym.make('Taxi-v2')
    obs_dim = env.observation_space.n
    model = TwoLayerDQN(obs_dim, 32, env.action_space.n)
    model.eval()

    eps = 0.01
    episodes = args.episodes
    cr = 0
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            if torch.rand(()) > eps:
                v_obs = make_state(obs_dim, torch.LongTensor([obs]))
                act = torch.argmax(model(v_obs)).item()
            else:
                act = env.action_space.sample()

            obs, r, done, _ = env.step(act)
            cr += r

    avg_r = cr * 1.0 / episodes


