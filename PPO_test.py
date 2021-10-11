import numpy as np
import pygame
from rocket_gym import RocketMeister10
from network import Net
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from torch.distributions import Beta

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (10)), ('a', np.float64, (2)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (10))])
env_config = {
    'gui': True,
    'env_name': 'default',
    # 'env_name': 'empty',
    # 'env_name': 'level1',
    # 'env_name': 'level2',
    # 'env_name': 'random',
    # 'camera_mode': 'centered',
    # 'env_flipped': False,
    # 'env_flipmode': False,
    # 'export_frames': True,
    'export_states': True,
    # 'export_highscore': False,
    'export_string': 'ppo',
    'max_steps': 10000,
    'gui_reward_total': True,
    'gui_echo_distances': True,
    'gui_level': True,
    'gui_velocity': True,
    'gui_goal_ang': True,
    'gui_frames_remaining': True,
    'gui_draw_echo_points': True,
    'gui_draw_echo_vectors': True,
    'gui_draw_goal_points': True,
}
class Agent():
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    buffer_capacity, batch_size = 3600, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0
        self.ppo_epoch = 10

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self,state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()

        del state
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1
        gamma = 0.99

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
        # Del from gpu to avoid overflow.
        del s, a, r, s_, old_a_logp
    def load_param(self,path= 'param/ppo_net_params.pkl'):
        print(path)
        self.net.load_state_dict(torch.load(path))

import pandas as pd
if __name__ == "__main__":
    render=True
    agent = Agent()
    agent.load_param('param/ppo_net_params_1.pkl')
    env = RocketMeister10()
    env.render()
    training_records = []
    running_score = 0
    best_score = 0
    df = pd.DataFrame()
    state = env.reset()
    for i_ep in range(10): # change the values if you want to test more than 1 time
        score = 0
        state = env.reset()

        for t in range(10000):
            action,_ = agent.select_action(state)
            state_ , reward,done,_ = env.step(action * np.array([0.1,2.]) + np.array([-0.05, -1.]))
            if render:
                env.render() 
            score += reward
            state = state_
            if env.rocket.collision:
                break

        print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        env.env.close()