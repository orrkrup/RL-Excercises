#!/usr/bin/env python

import os
import sys

import torch
from torch import nn
import gym
import torch.nn.functional as F
import datetime
import time
import numpy as np
from tqdm import trange
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from taxi_dqn.train_dqn import lineplotCI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BatchEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        for ind, env in enumerate(self.env_batch):
            obs, reward, done, _ = env.step(action_n[ind])
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n.append(reward)
            done_n.append(done)
            if done:
                obs = env.reset()
            obs_n.append(obs)

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n.append(env.reset())
        return obs_n

    # render environment
    def _render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

    def rand_act(self):
        return [env.action_space.sample() for env in self.env_batch]


class TwoLayerFC(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(TwoLayerFC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, x_in):
        return self.net(x_in)


class PGModel(nn.Module):
    def __init__(self, in_dim, h_dim, act_dim):
        super(PGModel, self).__init__()
        self.p = TwoLayerFC(in_dim, h_dim, act_dim)
        self.v = TwoLayerFC(in_dim, h_dim, 1)

    def forward(self, x_in):
        return self.p(x_in), self.v(x_in)


class PGLoss(object):
    def __init__(self, gamma, lamb, v_weighting, eps, eps_delta, eps_end):
        self.gamma = gamma
        self.lamb = lamb
        self.vc = v_weighting
        self.ec = eps
        self.ec_delta = eps_delta
        self.ec_end = eps_end
        # self.mse = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def calc_loss(self, steps, v_final):

        returns = [v_final.squeeze()]
        next_values = v_final.squeeze()
        advantages = [torch.zeros(v_final.size(0), device=device)]
        for s in steps:
            rewards, dones, actions, policies, values = [obj.to(device) for obj in s]
            w = self.gamma * dones
            returns.append(w * returns[-1] + rewards)
            errors = rewards + w * next_values - values.squeeze()
            advantages.append(w * advantages[-1] * self.lamb + errors)
            next_values = values.squeeze()

        _, _, actions, policies, values = [torch.stack(obj, dim=1) for obj in zip(*steps)]

        probs = F.softmax(policies, dim=-1)
        log_probs = F.log_softmax(policies, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.long().to(device).unsqueeze(-1))

        returns = torch.stack(returns[1:], dim=1).unsqueeze(-1)
        advantages = torch.stack(advantages[1:], dim=1).unsqueeze(-1)

        p_loss = - torch.sum(action_log_probs * advantages.detach())
        v_loss = self.mse(values, returns.detach())
        e_loss = torch.sum(log_probs * probs)

        if self.ec > self.ec_end:
            self.ec -= self.ec_delta
        return p_loss + self.vc * v_loss + self.ec * e_loss


def to_categorical(inds, dim):
    v = torch.zeros((len(inds), dim))
    v[list(range(len(inds))), inds] = 1.0
    return v


def make_state(obs_dim, inds):
    v = to_categorical(inds, obs_dim)
    return v.to(device)


def train_model(opts):
    # Initialize Environment
    env_b = [gym.make('Taxi-v2') for _ in range(opts['num_workers'])]
    env = BatchEnv(env_b)

    obs_dim = env.observation_space.n

    # Initizlize Model, loss function and optimizer
    net = PGModel(obs_dim, opts['net_h_dim'], env.action_space.n).to(device)

    eps_delta = (opts['eps0'] - opts['eps_end']) / opts['eps_decay_steps']
    loss_obj = PGLoss(opts['gamma'], opts['lamb'], opts['value_weighting'], opts['eps0'], eps_delta, opts['eps_end'])
    loss_func = loss_obj.calc_loss
    opt = torch.optim.Adam(net.parameters(), lr=opts['lr'])
    opt.zero_grad()

    # Initialize other parameters:
    step_counter = 0
    eval_rewards = [0] * opts['num_workers']
    total_steps_plt = []
    reward_plt = []
    vis = Visdom(env='pg_taxi')
    vobs = env.reset()
    obs = make_state(obs_dim, torch.LongTensor(vobs))
    for _ in trange(1, int(opts['max_steps']), opts['num_workers'] * opts['rollout_steps']):
        # while step_counter < opts['max_steps']:
        steps = []
        for _ in range(opts['rollout_steps']):
            step_counter += opts['num_workers']
            # env.render()

            p, v = net(obs)

            try:
                act = [a.item() for a in torch.multinomial(F.softmax(p, dim=-1), 1)]
            except RuntimeError:
                print(p)
                raise

            vobs, r, done, _ = env.step(act)
            obs = make_state(obs_dim, torch.LongTensor(vobs))

            done_vec = [0.0 if (d and r[i] > 0) else 1.0 for i, d in enumerate(done)]
            steps.insert(0, (torch.Tensor(r), torch.Tensor(done_vec), torch.Tensor(act), p, v))

            eval_rewards = [eval_rewards[i] + rr for i, rr in enumerate(r)]
            for i, d in enumerate(done):
                if d:
                    total_steps_plt.append(step_counter)
                    reward_plt.append(eval_rewards[i])
                    eval_rewards[i] = 0

                    d = dict(title='Evaluated Reward', xlabel='Total Steps', ylabel='Average Reward')
                    vis.line(Y=reward_plt, X=total_steps_plt, win='eval_r', opts=d)

        _, v_final = net(obs)

        loss = loss_func(steps, v_final)
        loss.backward()
        opt.step()
        opt.zero_grad()

    torch.save(net.state_dict(), opts['save_path'])
    env.close()
    return reward_plt, total_steps_plt


def load_and_plot(filename, smooth=False):
    import scipy.stats as stats

    opts, runs_x, runs_y = torch.load(filename)
    vis = Visdom(env='dqn_taxi')

    if smooth:
        zeros = torch.zeros(runs_y.shape)
        smooth_runs = [torch.cat((zeros[:, -n:], runs_y[:, :-n]), dim=1).unsqueeze(-1) for n in range(1, 200)]
        smooth_runs = torch.cat([runs_y.unsqueeze(-1)] + smooth_runs, dim=-1).squeeze().transpose(1,0)
        # smooth_runs = torch.mean(smooth_runs, dim=-1)
        line_lbs = []
        line_ubs = []

    line_lb = []
    line_ub = []
    if len(runs_y.shape) != 1:
        if smooth:
            for ind in range(smooth_runs.shape[1]):
                min_line, max_line = stats.t.interval(0.95, len(smooth_runs[:, ind]) - 1,
                                                      loc=torch.mean(smooth_runs[:, ind]),
                                                      scale=stats.sem(smooth_runs[:, ind]))
                line_lbs.append(min_line)
                line_ubs.append(max_line)
            lineplotCI(line=torch.mean(smooth_runs, dim=0), line_lb=line_lbs, line_ub=line_ubs, name=filename[:-4],
                       x=runs_x.squeeze())
        else:
            for ind in range(runs_y.shape[1]):
                min_line, max_line = stats.t.interval(0.95, len(runs_y[:, ind]) - 1,
                                                      loc=torch.mean(runs_y[:, ind]),
                                                      scale=stats.sem(runs_y[:, ind]))
                line_lb.append(min_line)
                line_ub.append(max_line)
            lineplotCI(line=torch.mean(runs_y, dim=0), line_lb=line_lb, line_ub=line_ub, name=filename[:-4],
                       x=runs_x.sqeeze())
    # vis.update_window_opts(win='eval_r', opts=dict(legend=[str(s) for s in hidden_sizes]))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from visdom import Visdom

    # dirlist = os.listdir('./')
    # pkl_list = [i for i in dirlist if i[:8] == 'eps0_1.0']
    # for filename in pkl_list:
    #     load_and_plot(filename, smooth=True)
    # # plt.legend()
    # plt.show()
    # exit()

    # parse args
    opts = {
        'net_h_dim': 50,
        'eps0': 1.0,
        'gamma': 0.99,
        'eps_decay_steps': 1e6,
        'eps_end': 0.0,
        'lr': 0.01,
        'save_path': './results/model_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'),
        'number_of_runs': 1,
        'num_workers': 64,
        'lamb': 0.95,
        'value_weighting': 0.5,
        'max_steps': 1.5e6,
        'rollout_steps': 40
    }
    vis = Visdom(env='dqn_taxi')

    for eps0 in [1.0]:
        opts['eps0'] = eps0
        print('eps0 = ' + str(eps0))

        runs_x = []
        runs_y = []
        line_lb = []
        line_ub = []
        for val in range(opts['number_of_runs']):
            opts['save_path'] = './results/model_' + datetime.datetime.now().strftime('%Y%m%d_%H%M')

            st = time.time()
            reward_plt, total_steps_plt = train_model(opts)

            # d = dict(title='Evaluated Reward', xlabel='Evaluation Epochs', ylabel='Average Reward')
            # x = list(range(1, len(eval_rewards) + 1))
            # vis.line(Y=eval_rewards, X=x, win='eval_r', opts=d, update='append', name=str(val))

            et = time.time() - st
            print('Run number {}. Took {} seconds'.format(val, et))

            runs_x.append(total_steps_plt)
            runs_y.append(reward_plt)

        runs_x = torch.Tensor(runs_x)
        runs_y = torch.Tensor(runs_y)
        # if len(runs.shape) != 1:
        #     for ind in range(runs.shape[1]):
        #         min_line, max_line = stats.t.interval(0.95, len(runs[:, ind]) - 1,
        #                                                               loc=torch.mean(runs[:, ind]),
        #                                                               scale=stats.sem(runs[:, ind]))
        #         line_lb.append(min_line)
        #         line_ub.append(max_line)
        #     lineplotCI(line=torch.mean(runs, dim=0), line_lb=line_lb, line_ub=line_ub)
        # vis.update_window_opts(win='eval_r', opts=dict(legend=[str(s) for s in hidden_sizes]))
        name = 'eps0_' + str(opts['eps0']) + '.pkl'
        torch.save((opts, runs_x, runs_y), name)
