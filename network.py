import torch
import torch.nn as nn
import torch.optim as optimizer
import argparse
import torch.nn.functional as F
from torch.distributions import Beta 



import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, downsample=None):
        super(Net, self).__init__()
        self.fc1= nn.Linear(10, 100)
        self.v = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 1))
        self.fc = nn.Sequential(nn.Linear(100, 200), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(200, 2), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(200, 2), nn.Softplus())
        self.conv_drop = nn.Dropout2d()
        self.apply(self._weights_init)
        self.downsample=downsample

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        residual = x
        if(self.downsample is not None):
            residual = self.downsample(residual)
        x = F.relu(F.max_pool1d(self.conv_drop(self.fc1(x)).view(-1,1,100),kernel_size=1))+ residual.view(-1,1,100)
        x = x.view(-1, 100)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1 

        return (alpha, beta), v

class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width,num_layers=2,RNN=False):
		super(BetaActor, self).__init__()
		self.num_layers=num_layers
		self.net_width =net_width
		self.state_dim=state_dim
		self.RNN=RNN
		if self.RNN:
			self.rnn= nn.RNN(state_dim,net_width,num_layers,nonlinearity="tanh",batch_first=False)
		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)
		self.apply(self._weights_init)
	
	@staticmethod
	def _weights_init(m):
		if isinstance(m, nn.Conv1d):
			nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
			nn.init.constant_(m.bias, 0.1)


	def forward(self, state):
		if self.RNN:
			state = state.unsqueeze(0)
			h0=torch.zeros(self.num_layers,state.size(1),self.net_width).to(device)
			out,_ = self.rnn(state,h0)

			out = out[:,-1,:]
			alpha = F.softplus(self.alpha_head(out))+1.0
			beta = F.softplus(self.beta_head(out))+1.0
			return alpha,beta
		else:
			a = torch.tanh(self.l1(state))
			a = torch.tanh(self.l2(a))

			alpha = F.softplus(self.alpha_head(a)) + 1.0
			beta = F.softplus(self.beta_head(a)) + 1.0

			return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def dist_mode(self,state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v