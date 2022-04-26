import os
import math
import torch
import numpy as np
import torch.optim as optim

from models.ppo_network import ActorC, CriticC, ActorD, CriticD
from config import PPOConfig
ppo_config = PPOConfig()


class PPO:
    def __init__(self):
        self.actor_rc = ActorC(ppo_config.observation_dim, ppo_config.rc_action_dim).cuda()
        self.critic_rc = CriticC(ppo_config.observation_dim).cuda()
        self.actor_rc_optim = optim.Adam(self.actor_rc.parameters(), lr=ppo_config.actor_lr)
        self.critic_rc_optim = optim.Adam(self.critic_rc.parameters(), lr=ppo_config.critic_lr, weight_decay=ppo_config.weight_decay)

        self.actor_hf = ActorD(ppo_config.observation_dim, ppo_config.hf_action_dim).cuda()
        self.critic_hf = CriticD(ppo_config.observation_dim).cuda()
        self.actor_hf_optim = optim.Adam(self.actor_hf.parameters(), lr=ppo_config.actor_lr)
        self.critic_hf_optim = optim.Adam(self.critic_hf.parameters(), lr=ppo_config.critic_lr, weight_decay=ppo_config.weight_decay)

        self.criterion = torch.nn.MSELoss()

    def set_ac_mode(self, mode=True):
        self.actor_rc.train(mode)
        self.critic_rc.train(mode)
        self.actor_hf.train(mode)
        self.critic_hf.train(mode)

    def select_action_rc(self, state):
        mu, std, _ = self.actor_rc(state)
        action = torch.normal(mu, std).data.cpu().numpy()[0]
        return action

    def select_action_hf(self, state):
        dist = self.actor_hf(state)
        action = dist.sample()
        return action.item()

    def log_density(self, x, mu, std, log_std):
        """
        求正态分布的概率密度函数的log，把正态分布的概率密度函数外面套一个log然后化简就是本函数
        :param x: action
        :param mu: \mu
        :param std: \sigma
        :param log_std: log\sigma
        :return: the probability of action x happened in Normal(mu, std)
        """
        var = std.pow(2)
        log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std

        return log_density.sum(1, keepdim=True)

    def learn(self, memory_rc, memory_hf):
        memory_rc = np.array(memory_rc)
        states_rc = torch.Tensor(np.vstack(memory_rc[:, 0])).cuda()
        actions_rc = torch.Tensor(list(memory_rc[:, 1])).cuda()
        rewards_rc = torch.Tensor(list(memory_rc[:, 2])).cuda()
        masks_rc = torch.Tensor(list(memory_rc[:, 3])).cuda()
        # print(states_rc.shape, actions_rc.shape, rewards_rc.shape, masks_rc.shape)
        memory_hf = np.array(memory_hf)
        states_hf = torch.Tensor(np.vstack(memory_hf[:, 0])).cuda()
        actions_hf = torch.Tensor(list(memory_hf[:, 1])).cuda()
        rewards_hf = torch.Tensor(list(memory_hf[:, 2])).cuda()
        masks_hf = torch.Tensor(list(memory_hf[:, 3])).cuda()
        # print(states_hf.shape, actions_hf.shape, rewards_hf.shape, masks_hf.shape)

        values_rc = self.critic_rc(states_rc).squeeze(1)
        values_hf = self.critic_hf(states_hf).squeeze(1)

        # get_gae
        targets_rc, targets_hf, advants_rc, advants_hf = self.get_gae(rewards_rc, rewards_hf, values_rc, values_hf)

        mu_rc, std_rc, log_std_rc = self.actor_rc(states_rc)
        old_policy_rc = self.log_density(actions_rc, mu_rc, std_rc, log_std_rc).detach()

        dist_hf = self.actor_hf(states_hf)
        old_policy_hf = dist_hf.log_prob(actions_hf).detach()

        n = len(states_rc)
        arr = np.arange(n)
        for epoch in range(ppo_config.k_epoch):
            np.random.shuffle(arr)
            for i in range(n // ppo_config.batch_size):
                batch_index = torch.LongTensor(arr[ppo_config.batch_size * i: ppo_config.batch_size * (i + 1)])

                # rc part learn
                states_rc_samples = states_rc[batch_index]
                targets_rc_samples = targets_rc.unsqueeze(1)[batch_index]
                advants_rc_samples = advants_rc.unsqueeze(1)[batch_index]
                actions_rc_samples = actions_rc[batch_index]
                old_policy_rc_samples = old_policy_rc[batch_index]

                mu_rc, std_rc, log_std_rc = self.actor_rc(states_rc_samples)
                new_policy = self.log_density(actions_rc_samples, mu_rc, std_rc, log_std_rc)
                ratio = torch.exp(new_policy - old_policy_rc_samples)
                surrogate = ratio * advants_rc_samples
                clipped_ratio = torch.clamp(ratio, 1.0 - ppo_config.clip_param, 1.0 + ppo_config.clip_param)
                clipped_loss = clipped_ratio * advants_rc_samples

                actor_loss = -torch.min(surrogate, clipped_loss).mean()
                self.actor_rc_optim.zero_grad()
                actor_loss.backward()
                self.actor_rc_optim.step()

                values = self.critic_rc(states_rc_samples)
                critic_loss = self.criterion(values, targets_rc_samples)
                self.critic_rc_optim.zero_grad()
                critic_loss.backward()
                self.critic_rc_optim.step()

                # hf part learn
                states_hf_samples = states_hf[batch_index]
                targets_hf_samples = targets_hf.unsqueeze(1)[batch_index]
                advants_hf_samples = advants_hf.unsqueeze(1)[batch_index]
                actions_hf_samples = actions_hf[batch_index]
                old_policy_hf_samples = old_policy_hf[batch_index]

                dist_hf = self.actor_hf(states_hf_samples)
                new_policy = dist_hf.log_prob(actions_hf_samples)
                ratio = torch.exp(new_policy - old_policy_hf_samples)
                surrogate = ratio * advants_hf_samples
                clipped_ratio = torch.clamp(ratio, 1.0 - ppo_config.clip_param, 1.0 + ppo_config.clip_param)
                clipped_loss = clipped_ratio * advants_rc_samples

                actor_loss = -torch.min(surrogate, clipped_loss).mean()
                self.actor_hf_optim.zero_grad()
                actor_loss.backward()
                self.actor_hf_optim.step()

                values = self.critic_hf(states_hf_samples)
                critic_loss = self.criterion(values, targets_hf_samples)
                self.critic_hf_optim.zero_grad()
                critic_loss.backward()
                self.critic_hf_optim.step()

    def get_gae(self, rewards_rc, rewards_hf, values_rc, values_hf):
        targets_hf = rewards_hf
        targets_rc = rewards_rc.cuda() + ppo_config.gamma*targets_hf

        td_error_hf = rewards_hf - values_hf.data
        td_error_rc = rewards_rc + ppo_config.gamma*td_error_hf - values_rc.data

        advants_hf = td_error_hf
        advants_hf = (advants_hf - advants_hf.mean())/advants_hf.std()
        advants_rc = td_error_rc + ppo_config.gamma*ppo_config.lamda*td_error_hf
        advants_rc = (advants_rc - advants_rc.mean())/advants_rc.std()

        return targets_rc, targets_hf, advants_rc, advants_hf

    def save_model(self, epoch):
        if not os.path.exists("saved/CMC_PPO/models/epoch-" + str(epoch)):
            os.mkdir("saved/CMC_PPO/models/epoch-" + str(epoch))
        torch.save(self.actor_rc.state_dict(), "saved/CMC_PPO/models/epoch-" + str(epoch) + "/actor_rc.pth")
        torch.save(self.critic_rc.state_dict(), "saved/CMC_PPO/models/epoch-" + str(epoch) + "/critic_rc.pth")
        torch.save(self.actor_hf.state_dict(), "saved/CMC_PPO/models/epoch-" + str(epoch) + "/actor_hf.pth")
        torch.save(self.critic_hf.state_dict(), "saved/CMC_PPO/models/epoch-" + str(epoch) + "/critic_hf.pth")
