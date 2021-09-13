import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))
Transition_w_na = namedtuple('Transition_w_na', ('state', 'next_state', 'action', 'next_action', 'reward', 'mask'))


class IQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, quantile_embedding_dim, num_quantile_sample, device, env_name=None, fc1_units=64, fc2_units=128):
        super(IQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device
        self.env_name = env_name
        if env_name.__contains__("CartPole"):
            fc1_units = 128
            self.fc1 = nn.Linear(num_inputs, fc1_units)
            self.fc2 = nn.Linear(fc1_units, num_outputs)
            self.phi = nn.Linear(self.quantile_embedding_dim, fc1_units)
        else:
            self.fc1 = nn.Linear(num_inputs, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, num_outputs)
            self.phi = nn.Linear(self.quantile_embedding_dim, fc2_units)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles, self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx)

        phi = self.phi(cos_tau)
        phi = F.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.num_inputs)
        state_tile = state_tile.flatten().view(-1, self.num_inputs)

        x = F.relu(self.fc1(state_tile))
        if self.env_name.__contains__("CartPole"):
            x = self.fc2(x * phi)
        else:
            x = F.relu(self.fc2(x))
            x = self.fc3(x * phi)
        z = x.view(-1, num_quantiles, self.num_outputs)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, batch_size, num_tau_sample, device, num_tau_prime_sample, gamma):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        z = online_net(states, tau, num_tau_sample)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action.to(device)).squeeze(1)

        tau_prime = torch.Tensor(np.random.rand(batch_size * num_tau_prime_sample, 1))
        next_z = target_net(next_states, tau_prime, num_tau_prime_sample)
        next_action = next_z.mean(dim=2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_tau_prime_sample)
        next_z_a = next_z.gather(1, next_action).squeeze(1)

        T_z = rewards.to(device).unsqueeze(1) + gamma * next_z_a * masks.to(device).unsqueeze(1)

        T_z_tile = T_z.view(-1, num_tau_prime_sample, 1).expand(-1, num_tau_prime_sample, num_tau_sample)
        z_a_tile = z_a.view(-1, 1, num_tau_sample).expand(-1, num_tau_prime_sample, num_tau_sample)

        error_loss = T_z_tile - z_a_tile
        huber_loss = F.smooth_l1_loss(z_a_tile, T_z_tile.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @classmethod
    def eval_model(cls, online_net, target_net, batch, batch_size, num_tau_sample, device, num_tau_prime_sample, gamma):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        z = online_net(states, tau, num_tau_sample)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action.to(device)).squeeze(1)

        tau_prime = torch.Tensor(np.random.rand(batch_size * num_tau_prime_sample, 1))
        next_z = target_net(next_states, tau_prime, num_tau_prime_sample)
        next_action = next_z.mean(dim=2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_tau_prime_sample)
        next_z_a = next_z.gather(1, next_action).squeeze(1)

        T_z = rewards.to(device).unsqueeze(1) + gamma * next_z_a * masks.to(device).unsqueeze(1)

        T_z_tile = T_z.view(-1, num_tau_prime_sample, 1).expand(-1, num_tau_prime_sample, num_tau_sample)
        z_a_tile = z_a.view(-1, 1, num_tau_sample).expand(-1, num_tau_prime_sample, num_tau_sample)

        error_loss = T_z_tile - z_a_tile
        huber_loss = F.smooth_l1_loss(z_a_tile, T_z_tile.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()
        return loss

    def get_action(self, state, num_quantile_sample):
        tau = torch.Tensor(np.random.rand(num_quantile_sample, 1) * 0.5)  # CVaR
        z = self.forward(state, tau, num_quantile_sample)
        q = z.mean(dim=2, keepdim=True)
        action = torch.argmax(q)
        return action.item(), z


class RecurrentIQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, gru_size, quantile_embedding_dim, num_quantile_sample, device, fc1_units=64):
        super(RecurrentIQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.gru_size = gru_size
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device

        self.gru = nn.GRUCell(num_inputs, gru_size)
        self.gru_fc = nn.Linear(gru_size, fc1_units)
        self.fc = nn.Linear(fc1_units, num_outputs)

        self.phi = nn.Linear(self.quantile_embedding_dim, 64)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, hx, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles, self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx).to(self.device)

        phi = self.phi(cos_tau)
        phi = F.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.num_inputs)
        state_tile = state_tile.flatten().view(-1, self.num_inputs).to(self.device)

        ghx = self.gru(state_tile, hx)
        x = self.gru_fc(ghx)
        x = self.fc(x * phi)

        z = x.view(-1, num_quantiles, self.num_outputs)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z, ghx

    @classmethod
    def train_model(cls, model, optimizer, hx, states, actions, target, batch_size, num_tau_sample, device):
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        states = states.reshape(states.shape[0], 1, -1)
        z, hx = model(states, hx, tau, num_tau_sample)
        action = actions.long().unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action.to(device)).squeeze(1)

        T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample)

        error_loss = T_z - z_a
        huber_loss = F.smooth_l1_loss(z_a, T_z.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, hx

    @classmethod
    def eval_model(cls, model, hx, states, actions, target, batch_size, num_tau_sample, device):
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        states = states.reshape(states.shape[0], 1, -1)
        z, hx = model(states, hx, tau, num_tau_sample)
        action = actions.long().unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action.to(device)).squeeze(1)

        T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample)

        error_loss = T_z - z_a
        huber_loss = F.smooth_l1_loss(z_a, T_z.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.sum(dim=1).mean()
        return loss, hx


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class Memory_w_na(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, next_action, reward, mask):
        self.memory.append(Transition_w_na(state, next_state, action, next_action, reward, mask))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition_w_na(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


def get_action(state, target_net, epsilon, env, num_quantile_sample):
    if np.random.rand() <= epsilon:
        return env.action_space.sample(), None
    else:
        action, z = target_net.get_action(state, num_quantile_sample)
        return action, z


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())
