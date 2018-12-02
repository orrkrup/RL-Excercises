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
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvolutionalDQN(nn.Module):
    def __init__(self, act_dim, dropout=0):
        super(ConvolutionalDQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, act_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


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


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
    return resize(screen).unsqueeze(0).to(device)


def to_categorical(inds, dim):
    v = torch.zeros((len(inds), dim))
    v[list(range(len(inds))), inds] = 1.0
    return v


def make_state(obs_dim, inds):
    v = to_categorical(inds, obs_dim)
    return v


def eval_model(model, obs_dim, env):
    model.eval()
    eps = 0.01
    episodes = 20
    cr = 0
    for ep in range(episodes):
        _ = env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        obs = current_screen - last_screen

        done = False
        while not done:
            if torch.rand(()) > eps:
                act = torch.argmax(model(obs)).item()
            else:
                act = env.action_space.sample()

            _, r, done, _ = env.step(act)

            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                obs = current_screen - last_screen
            else:
                obs = None

            cr += r

    avg_r = cr * 1.0 / episodes
    model.train()
    return avg_r


def train_model(opts):
    # Initialize Environment
    env = gym.make('Acrobot-v1')

    # Initialize Experience Reply
    er = ExperienceReplay(opts['max_er'])

    if opts['dense_obs']:
        print('No dense representation')
    else:
        obs_dim = None

    # Initizlize Model, loss function and optimizer
    net = ConvolutionalDQN(env.action_space.n, opts['dropout']).to(device)
    loss_obj = DQNLoss(net, er, opts['gamma'], opts['minibatch_size'], obs_dim, opts['l1_reg'])
    loss_func = loss_obj.calc_loss
    opt = torch.optim.Adam(net.parameters(), lr=opts['lr'])

    # Initialize other parameters:
    eps = opts['eps0']
    step_counter = 0
    eval_rewards = []
    # vis = Visdom(env='dqn_taxi')
    for e in trange(opts['episodes']):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        obs = current_screen - last_screen

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

            _, r, done, _ = env.step(act)
            cr += r

            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                obs = current_screen - last_screen
            else:
                obs = None

            if not (done and r < 0):
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

        # print("Episode {} done. Reward: {}".format(e, cr))

    # torch.save(net.state_dict(), opts['save_path'])
    return eval_rewards


def lineplotCI(line, line_lb, line_ub):
    # plot the data
    x = range(1, line.shape[0] + 1)
    plt.figure(1)
    # plot the shaded range of the confidence intervals
    if line_ub is not None and line_lb is not None:
        plt.fill_between(np.array(x), np.array(line_ub), np.array(line_lb), alpha=.5)
    # plot the mean on top
    plt.plot(np.array(x), np.array(line))
    plt.grid()
    plt.show()


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
        'dense_obs': False,
        'number_of_runs': 1
    }
    vis = Visdom(env='dqn_taxi')

    runs = []
    line_lb = []
    line_ub = []
    hidden_sizes = [64]
    for val in range(opts['number_of_runs']):
        st = time.time()
        eval_rewards = train_model(opts)

        d = dict(title='Evaluated Reward', xlabel='Evaluation Epochs', ylabel='Average Reward')
        x = list(range(1, len(eval_rewards) + 1))
        vis.line(Y=eval_rewards, X=x, win='eval_r', opts=d, update='append', name=str(val))

        et = time.time() - st
        print('Run number {}. Took {} seconds'.format(val, et))

        runs.append(eval_rewards)

    runs = torch.Tensor(runs)
    if len(runs.shape) != 1:
        for ind in range(runs.shape[1]):
            act_acc_min_line, act_acc_max_line = stats.t.interval(0.95, len(runs[:, ind]) - 1,
                                                                  loc=torch.mean(runs[:, ind]),
                                                                  scale=stats.sem(runs[:, ind]))
            line_lb.append(act_acc_min_line)
            line_ub.append(act_acc_max_line)
        lineplotCI(line=torch.mean(runs, dim=0), line_lb=line_lb, line_ub=line_ub)
    vis.update_window_opts(win='eval_r', opts=dict(legend=[str(s) for s in hidden_sizes]))
