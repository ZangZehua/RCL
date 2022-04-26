import torch
import torch.nn as nn
from torch.distributions import Categorical


from config import PPONetworkConfig
net_config = PPONetworkConfig()


class ActorC(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, net_config.hidden_dim_1),
            nn.GELU(),
            nn.Linear(net_config.hidden_dim_1, net_config.hidden_dim_2),
            nn.GELU()
        )
        self.fc_last = nn.Linear(net_config.hidden_dim_2, action_dim)
        # init fc_last (for boost training?)
        # self.fc_last.weight.data.mul_(0.1)
        # self.fc_last.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc(x)
        mu = self.fc_last(x)  # get \mu
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)  # get \std
        return mu, std, log_std


class CriticC(nn.Module):
    def __init__(self, input_dim):
        super(CriticC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, net_config.hidden_dim_1),
            nn.GELU(),
            nn.Linear(net_config.hidden_dim_1, net_config.hidden_dim_2)
        )
        self.fc_last = nn.Linear(net_config.hidden_dim_2, 1)
        # same with actor
        # self.fc_last.weight.data.mul_(0.1)
        # self.fc_last.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc(x)
        v = self.fc_last(x)
        return v


class ActorD(nn.Module):
    def __init__(self, input_dim, action_n):
        super(ActorD, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, net_config.hidden_dim_1),
            nn.GELU(),
            nn.Linear(net_config.hidden_dim_1, net_config.hidden_dim_2),
            nn.GELU(),
            nn.Linear(net_config.hidden_dim_2, action_n),
            nn.Softmax(dim=-1)
        )
        # self.fc_last =
        # # init fc_last (for boost training?)
        # self.fc_last.weight.data.mul_(0.1)
        # self.fc_last.bias.data.mul_(0.0)

    def forward(self, x):
        action_probs = self.fc(x)
        dist = Categorical(action_probs)
        return dist


class CriticD(nn.Module):
    def __init__(self, input_dim):
        super(CriticD, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, net_config.hidden_dim_1),
            nn.GELU(),
            nn.Linear(net_config.hidden_dim_1, net_config.hidden_dim_2),
            nn.GELU(),
            nn.Linear(net_config.hidden_dim_2, 1),
        )

    def forward(self, x):
        v = self.fc(x)
        return v

