#!/usr/bin/env python


import torch
from torch import nn
import copy
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TwoLayerDQN(nn.Module):
    def __init__(self, in_dim, h_dim, act_dim):
        super(TwoLayerDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, act_dim)
        )

    def forward(self, x_in):
        return self.net(x_in)


class DQNLoss(object):
    def __init__(self, net):
        self.net = copy.deepcopy(net)

    def calc_loss(self):
        # TODO: implement DQN loss
        pass


class ExpereinceReplay(object):
    def __init__(self, max_size):
        self.len = max_size
        self.q = []

    def push(self, obj):
        self.q.insert(0, obj)
        if len(self.q) > self.len:
            q.pop()


def train_model(opts):

    # Initialize Environment
    env = gym.make('Taxi-v2')

    # Initialize Experience Reply
    er = ExpereinceReplay(opts['max_er'])

    # Initizlize Model, loss function and optimizer
    net = TwoLayerDQN(env.observation_space.n, opts['net_h_dim'], env.action_space.n)
    loss_func = DQNLoss(net).calc_loss
    opt = torch.optim.Adam(net.parameters())

    # Initialize other parameters:
    eps = opts['eps0']
    for e in range(opts['episodes']):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            last_obs = obs

            # w. p. epsilon:
            # act = torch.argmax(net(obs))
            # else
            act = env.action_space.sample()
            obs, r, done, _ = env.step(act)
            er.push((last_obs, act, r, obs))


if __name__ == '__main__':

    # parse args
    opts = {
        'episodes': 20,
        'max_er': 1e4,
        'net_h_dim': 64,
        'eps0': 1,
        'eps_decay': 0.99,
    }
    train_model(opts)