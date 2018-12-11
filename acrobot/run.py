#!/usr/bin/env python

import argparse
import torch
import gym
from tqdm import trange
import imageio

from dqn_acrobot import ConvolutionalDQN, get_screen_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--modelfilename', type=str, default='./acrobot/opt_model', help='Filename for input '
                                                                                               'model')
    parser.add_argument('-s', '--save_video', type=int, default=0, help='Save GIF of model running for the number of '
                                                                        'episodes specified in this option')
    parser.add_argument('-e', '--episodes', type=int, default=100, help='How many episodes to run model for')

    args = parser.parse_args()

    env = gym.make('Acrobot-v1')
    hist_len = 4
    model = ConvolutionalDQN(env.action_space.n, hist_len, dueling=True).cuda()
    model.load_state_dict(torch.load(args.modelfilename))
    model.eval()

    eps = 0.00
    episodes = args.episodes
    rewards = []
    cr = 0
    if args.save_video > 0:
        episodes = args.save_video
        images = []

    for _ in trange(episodes):
        done = False
        env.reset()
        screen, img = get_screen_(env)
        if args.save_video > 0:
            images.append(img)
        s = 1 - screen
        obs = torch.cat([s for _ in range(hist_len)], dim=1)

        while not done:
            if torch.rand(()) > eps:
                act = torch.argmax(model(obs)).item()
            else:
                act = env.action_space.sample()

            _, r, done, _ = env.step(act)

            screen, img = get_screen_(env)
            if args.save_video:
                images.append(img)
            obs = torch.cat([obs[:, 1:, :, :], 1 - screen], dim=1)

            cr += r

        rewards.append(cr)
        cr = 0

    env.close()

    r = torch.tensor(rewards)
    print("Model Evaluation done. Average reward: {} +- {}".format(torch.mean(r), torch.std(r)))

    if args.save_video > 0:
        imageio.mimsave('acrobot.gif', images)
