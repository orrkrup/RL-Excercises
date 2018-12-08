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
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, act_dim)

    def forward(self, x):
        x = self.net(x)
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
        minibatch = self.er.sample(self.batch_size)
        mb_s, mb_a, mb_r, mb_st, mb_t = zip(*minibatch)
        mb_s = torch.cat(mb_s, dim=0)
        mb_st = torch.cat(mb_st, dim=0)
        mb_t = torch.Tensor(mb_t).to(device)
        mb_a = torch.LongTensor(mb_a).to(device)
        mb_r = torch.Tensor(mb_r).to(device)
        with torch.no_grad():
            y = mb_r.float() + self.gamma * torch.max(self.target_net(mb_st), dim=-1)[0] * mb_t.float()
        Q = net(mb_s).gather(1, mb_a.unsqueeze(-1)).squeeze()
        l1_norm = torch.sum(torch.Tensor([torch.sum(torch.abs(param)) for param in net.parameters()]).to(device))
        return self.l(Q, y) + self.l1_reg * l1_norm


class ExperienceReplay(object):
    def __init__(self, max_size, success=False, suc_pr=0.5, suc_steps = 5e5):
        self.size = max_size
        self.q = []
        if success:
            self.suc_mem = ExperienceReplay(int(0.1 * max_size))
            self.suc_pr = suc_pr
            self.suc_delta = suc_pr / suc_steps
        else:
            self.suc_mem = None

    def push(self, obj, suc=False, suc_len=100):
        self.q.insert(0, obj)
        if len(self.q) > self.size:
            self.q.pop()
        if suc:
            self.suc_mem.push_batch_(self.q[:suc_len])

    def push_batch_(self, b):
        self.q = b + self.q
        if len(self.q) > self.size:
            self.q = self.q[:self.size]

    def sample(self, b_size):
        if self.suc_mem:
            self.suc_pr -= self.suc_delta

        if self.suc_mem is None or len(self.suc_mem) < b_size or torch.rand(()) > self.suc_pr:
            b_inds = torch.randperm(len(self))[:b_size]
            return self[b_inds]
        else:
            # Sample from success memory and reduce probability
            return self.suc_mem.sample(b_size)

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
                        T.Grayscale(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
    return resize(screen).unsqueeze(0).to(device)


def to_categorical(inds, dim):
    v = torch.zeros((len(inds), dim))
    v[list(range(len(inds))), inds] = 1.0
    return v


def eval_model(model, env):
    print("Evaluating model...")
    model.eval()
    eps = 0.01
    eval_episodes = 10
    cr = 0

    for _ in range(eval_episodes):
        done = False
        env.reset()
        s = 1 - get_screen(env)
        obs = torch.cat([s for _ in range(opts['hist_len'])], dim=1)

        while not done:
            if torch.rand(()) > eps:
                act = torch.argmax(model(obs)).item()
            else:
                act = env.action_space.sample()

            _, r, done, _ = env.step(act)
            obs = torch.cat([obs[:, 1:, :, :], 1 - get_screen(env)], dim=1)

            cr += r

    avg_r = cr / eval_episodes
    model.train()
    print("Evaluation done.")
    return avg_r


def train_model(opts):
    # Initialize Environment
    env = gym.make('Acrobot-v1')

    # Initialize Experience Reply
    er = ExperienceReplay(opts['max_er'], success=opts['success_exp_replay'])

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
    max_r = -1000
    vis = Visdom(env='dqn_acrobot')
    t = trange(opts['episodes'])
    declared = False
    eps_delta = (opts['eps0'] - opts['eps_end']) / opts['eps_decay_steps']
    loss_c = []
    for e in t:
        env.reset()
        # last_screen = get_screen(env)
        # current_screen = get_screen(env)
        # obs = current_screen - last_screen
        s = 1 - get_screen(env)
        obs = torch.cat([s for _ in range(opts['hist_len'])], dim=1)

        done = False
        cr = 0
        ep_start = step_counter
        while not done:
            step_counter += 1
            # env.render()
            last_obs = obs

            if torch.rand(()) > eps:
                act = torch.argmax(net(obs)).item()
            else:
                act = env.action_space.sample()

            _, r, done, _ = env.step(act)
            cr += r

            # last_screen = current_screen
            # current_screen = get_screen(env)
            # obs = current_screen - last_screen
            obs = torch.cat([obs[:, 1:, :, :], 1 - get_screen(env)], dim=1)
            # vis.image(torch.cat([obs[:, i, :, :] for i in range(4)], dim=-2), win='observation')

            if not (done and r < 0):
                er.push((last_obs, act, r, obs, 0 if done else 1), suc=(r > -1),
                        suc_len=min(step_counter - ep_start, opts['max_suc_len']))

            if step_counter > opts['learn_start_steps']:
                if not declared:
                    print("Learning started")
                    declared = True
                if eps > opts['eps_end']:
                    eps -= eps_delta
                opt.zero_grad()
                loss = loss_func(net)
                loss_c.append(loss.item())
                loss.backward()
                opt.step()

                # vis.line(Y=loss_c, X=list(range(len(loss_c))), win='loss', opts=dict(title='DQN Loss'))

                if not step_counter % opts['targ_update_steps']:
                    loss_obj.update_target(net)

        t.set_description('Reward: {}'.format(cr))

        if (not e % opts['eval_e_freq']) and step_counter > opts['learn_start_steps']:
            eval_rewards.append(eval_model(net, env))
            d = dict(title='Evaluated Reward', xlabel='Evaluation Epochs', ylabel='Average Reward')
            x = list(range(1, len(eval_rewards) + 1))
            vis.line(Y=eval_rewards, X=x, win='eval_r', opts=d)
            if eval_rewards[-1] > max_r:
                torch.save(net.state_dict(), opts['save_path'])

        # print("Episode {} done. Reward: {}".format(e, cr))

    torch.save(net.state_dict(), opts['save_path'] + '_final')
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
        'max_er': 2e5,
        'net_h_dim': 64,
        'eps0': 1,
        'gamma': 0.99,
        'eps_decay_steps': 5e5,
        'eps_end': 0.05,
        'targ_update_steps': 1.5e3,
        'learn_start_steps': 1e4,
        'lr': 0.001,
        'minibatch_size': 64,
        'dropout': 0,
        'l1_reg': 0,
        'eval_e_freq': 20,
        'save_path': './results/model_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'),
        'dense_obs': False,
        'number_of_runs': 1,
        'success_exp_replay': True,
        'max_suc_len': 100,
        'hist_len': 4
    }
    vis = Visdom(env='dqn_acrobot')

    runs = []
    line_lb = []
    line_ub = []
    hidden_sizes = [64]
    for val in range(opts['number_of_runs']):
        st = time.time()
        eval_rewards = train_model(opts)

        d = dict(title='Evaluated Reward', xlabel='Evaluation Epochs', ylabel='Average Reward')
        x = list(range(1, len(eval_rewards) + 1))
        vis.line(Y=eval_rewards, X=x, win='eval_r_final', opts=d, update='append', name=str(val))

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
