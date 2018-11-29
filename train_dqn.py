#!/usr/bin/env python


import torch
from torch import nn
import copy
import gym
from visdom import Visdom
import torch.nn.functional as F
import datetime
import time
from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TwoLayerDQN(nn.Module):
    def __init__(self, in_dim, h_dim, act_dim, dropout=0):
        super(TwoLayerDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(h_dim, act_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x_in):
        return self.net(x_in)


class DQNLoss(object):
    def __init__(self, net, er, gamma, mbatch_size, obs_dim, l1_reg=0):
        self.target_net = copy.deepcopy(net)
        self.er = er
        self.gamma = gamma
        self.batch_size = mbatch_size
        self.obs_dim = obs_dim
        self.l = F.smooth_l1_loss
        self.l1_reg = l1_reg

    def update_target(self, net):
        self.target_net = copy.deepcopy(net)

    def calc_loss(self, net):
        b_inds = torch.randperm(len(self.er))[:self.batch_size]
        minibatch = self.er[b_inds]
        mb_s, mb_a, mb_r, mb_st, mb_t = [torch.LongTensor(a) for a in zip(*minibatch)]
        mb_s = make_state(self.obs_dim, mb_s)
        mb_st = make_state(self.obs_dim, mb_st)
        with torch.no_grad():
            y = mb_r.float() + self.gamma * torch.max(self.target_net(mb_st), dim=-1)[0] * mb_t.float()
        Q = net(mb_s).gather(1, mb_a.unsqueeze(-1)).squeeze()
        l1_norm = torch.sum(torch.Tensor([torch.sum(torch.abs(param)) for param in net.parameters()]))
        return self.l(Q, y) + self.l1_reg * l1_norm


class ExperienceReplay(object):
    def __init__(self, max_size):
        self.size = max_size
        self.q = []

    def push(self, obj):
        self.q.insert(0, obj)
        if len(self.q) > self.size:
            self.q.pop()

    def __len__(self):
        return len(self.q)

    def __getitem__(self, item):
        return [self.q[i] for i in item]


def decode(i):
    a = to_categorical((torch.fmod(i, 4)).long(), 4)
    i = torch.div(i, 4).long()
    b = to_categorical((torch.fmod(i, 5)), 5)
    i = torch.div(i, 5).long()
    c = to_categorical((torch.fmod(i, 5)), 5)
    i = torch.div(i, 5).long()
    d = to_categorical(i, 5)
    # assert 0 <= i < 5
    return torch.cat([d, c, b, a], dim=1).float()


def to_categorical(inds, dim):
    v = torch.zeros((len(inds), dim))
    v[list(range(len(inds))), inds] = 1.0
    return v


def make_state(obs_dim, inds):
    if 19 == obs_dim:
        v = decode(inds)
    else:
        v = to_categorical(inds, obs_dim)
    return v


def eval_model(model, obs_dim, env):
    model.eval()
    eps = 0.01
    episodes = 20
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
    model.train()
    return avg_r


def train_model(opts):

    # Initialize Environment
    env = gym.make('Taxi-v2')

    # Initialize Experience Reply
    er = ExperienceReplay(opts['max_er'])

    if opts['dense_obs']:
        obs_dim = 19
    else:
        obs_dim = env.observation_space.n

    # Initizlize Model, loss function and optimizer
    net = TwoLayerDQN(obs_dim, opts['net_h_dim'], env.action_space.n, opts['dropout'])
    loss_obj = DQNLoss(net, er, opts['gamma'], opts['minibatch_size'], obs_dim, opts['l1_reg'])
    loss_func = loss_obj.calc_loss
    opt = torch.optim.Adam(net.parameters(), lr=opts['lr'])

    # Initialize other parameters:
    eps = opts['eps0']
    step_counter = 0
    eval_rewards = []
    # vis = Visdom(env='dqn_taxi')
    for e in trange(opts['episodes']):
        obs = env.reset()
        done = False
        cr = 0
        while not done:
            step_counter += 1
            # env.render()
            last_obs = obs

            if torch.rand(()) > eps:
                v_obs = make_state(obs_dim, torch.LongTensor([obs]))
                act = torch.argmax(net(v_obs)).item()
            else:
                act = env.action_space.sample()

            obs, r, done, _ = env.step(act)
            cr += r
            if not(done and r < 0):
                er.push((last_obs, act, r, obs, 0 if done else 1))

            if step_counter > opts['learn_start_steps']:
                if eps > opts['eps_end']:
                    eps *= opts['eps_decay']
                opt.zero_grad()
                loss = loss_func(net)
                loss.backward()
                opt.step()

                if not step_counter % opts['targ_update_steps']:
                    loss_obj.update_target(net)

        if not e % opts['eval_steps']:
            eval_rewards.append(eval_model(net, obs_dim, env))
            # d = dict(title='Evaluated Reward', xlabel='Evaluation Epochs', ylabel='Average Reward')
            # x = list(range(1, len(eval_rewards) + 1))
            # vis.line(Y=eval_rewards, X=x, win='eval_r', opts=d, update='append')

        print("Episode {} done. Reward: {}".format(e, cr))

    # torch.save(net.state_dict(), opts['save_path'])
    return eval_rewards


if __name__ == '__main__':
    # parse args
    opts = {
        'episodes': 3500,
        'max_er': 20000,
        'net_h_dim': 64,
        'eps0': 1,
        'gamma': 0.99,
        'eps_decay': 0.999,
        'eps_end': 0.1,
        'targ_update_steps': 600,
        'learn_start_steps': 1e4,
        'lr': 0.002,
        'minibatch_size': 256,
        'dropout': 0,
        'l1_reg': 0,
        'eval_steps': 50,
        'save_path': './results/model_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'),
        'dense_obs': True
    }
    vis = Visdom(env='dqn_taxi')

    hidden_sizes = [64]
    for val in hidden_sizes:
        opts['net_h_dim'] = val
        st = time.time()
        eval_rewards = train_model(opts)

        d = dict(title='Evaluated Reward', xlabel='Evaluation Epochs', ylabel='Average Reward')
        x = list(range(1, len(eval_rewards) + 1))
        vis.line(Y=eval_rewards, X=x, win='eval_r', opts=d, update='append', name=str(val))

        et = time.time() - st
        print('Done with hidden dim {}. Took {} seconds'.format(val, et))

    vis.update_window_opts(win='eval_r', opts=dict(legend=[str(s) for s in hidden_sizes]))