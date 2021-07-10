import os
import math
import json

import gym
import torch
import random
import sklearn
import argparse
import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from detecta import detect_cusum
from sklearn.metrics import roc_curve
import torch.nn.functional as functional
from sklearn.neighbors import NearestNeighbors
from matplotlib.animation import FuncAnimation
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sns.set()


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


class AutoregressiveRecurrentIQN(nn.Module):
    def __init__(self, feature_len, gru_size, quantile_embedding_dim, num_quantile_sample, device,
                 fc1_units=64, fc2_units=64):
        super(AutoregressiveRecurrentIQN, self).__init__()
        self.gru_size = gru_size
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device
        self.feature_len = feature_len
        self.fc_1 = nn.Linear(feature_len, fc1_units)
        self.fc_2 = nn.Linear(fc1_units, fc2_units)
        self.gru = nn.GRUCell(fc2_units, gru_size)
        self.fc_3 = nn.Linear(gru_size, feature_len)

        self.phi = nn.Linear(self.quantile_embedding_dim, 64)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, hx, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles,
                                                                                        self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx).to(self.device)

        phi = self.phi(cos_tau)
        phi = functional.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.feature_len)
        state_tile = state_tile.flatten().view(-1, self.feature_len).to(self.device)

        x = functional.relu(self.fc_1(state_tile))
        x = functional.relu(self.fc_2(x))
        ghx = self.gru(x, hx)
        x = self.fc_3(ghx * phi)
        z = x.view(-1, num_quantiles, self.feature_len)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z, ghx


class AutoregressiveRecurrentIQN_v2(nn.Module):
    def __init__(self, feature_len, gru_size, quantile_embedding_dim, num_quantile_sample, device, fc1_units=64):
        super(AutoregressiveRecurrentIQN_v2, self).__init__()
        self.gru_size = gru_size
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device
        self.feature_len = feature_len
        self.fc_1 = nn.Linear(feature_len, fc1_units)
        self.gru = nn.GRUCell(fc1_units, gru_size)
        self.dropout = nn.Dropout(p=0.2)
        self.fc_2 = nn.Linear(gru_size, gru_size)
        self.fc_3 = nn.Linear(gru_size, feature_len)

        self.phi = nn.Linear(self.quantile_embedding_dim, gru_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, hx, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles,
                                                                                        self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx).to(self.device)

        phi = self.phi(cos_tau)
        phi = functional.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.feature_len)
        state_tile = state_tile.flatten().view(-1, self.feature_len).to(self.device)

        x = functional.relu(self.fc_1(state_tile))
        ghx = self.gru(x, hx)
        x = self.dropout(ghx)
        x = x + functional.relu(self.fc_2(x))
        x = self.fc_3(x * phi)
        z = x.view(-1, num_quantiles, self.feature_len)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z, ghx


class AutoregressiveIQN(nn.Module):
    def __init__(self, feature_len, quantile_embedding_dim, num_quantile_sample, device, fc1_units=64, fc2_units=64):
        super(AutoregressiveIQN, self).__init__()
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device
        self.feature_len = feature_len
        self.fc_1 = nn.Linear(feature_len, fc1_units)
        self.fc_2 = nn.Linear(fc1_units, fc2_units)
        self.fc_3 = nn.Linear(fc2_units, feature_len)

        self.phi = nn.Linear(self.quantile_embedding_dim, 64)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles,
                                                                                        self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx).to(self.device)

        phi = self.phi(cos_tau)
        phi = functional.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.feature_len)
        state_tile = state_tile.flatten().view(-1, self.feature_len).to(self.device)

        x = functional.relu(self.fc_1(state_tile))
        x = functional.relu(self.fc_2(x))
        x = self.fc_3(x * phi)
        z = x.view(-1, num_quantiles, self.feature_len)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z


def get_action(state, target_net, epsilon, env, num_quantile_sample):
    if np.random.rand() <= epsilon:
        return env.action_space.sample(), None
    else:
        action, z = target_net.get_action(state, num_quantile_sample)
        return action, z


def train_model(model, optimizer, hx, states, target, batch_size, num_tau_sample, device, is_recurrent, clip_value, feature_len):
    tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
    states = states.reshape(states.shape[0], 1, -1)
    if is_recurrent:
        z, hx = model(states, hx, tau, num_tau_sample)
    else:
        z = model(states, tau, num_tau_sample)
    T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample, feature_len).transpose(1, 2)

    error_loss = T_z - z
    huber_loss = functional.smooth_l1_loss(z, T_z.detach(), reduction='none')
    if num_tau_sample == 1:
        tau = torch.arange(0, 1, 1 / 100).view(1, 100)
    else:
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

    loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    if clip_value is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()
    if is_recurrent:
        return z.squeeze(2), loss, hx
    else:
        return z.squeeze(2), loss


def test_model(model, hx, states, target, batch_size, num_tau_sample, device, is_recurrent, feature_len):
    tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
    states = states.reshape(states.shape[0], 1, -1)
    if is_recurrent:
        z, hx = model(states, hx, tau, num_tau_sample)
    else:
        z = model(states, tau, num_tau_sample)
    T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample, feature_len).transpose(1, 2)

    error_loss = T_z - z
    huber_loss = functional.smooth_l1_loss(z, T_z.detach(), reduction='none')
    if num_tau_sample == 1:
        tau = torch.arange(0, 1, 1 / 100).view(1, 100)
    else:
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

    loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
    loss = loss.mean()
    if is_recurrent:
        return z, loss, hx
    else:
        return z, loss


def feed_forward(model, hx, states, batch_size, num_tau_sample, sampling_size, is_recurrent, tree_root=False):
    states = states.reshape(states.shape[0], 1, -1)
    if tree_root:
        tau = torch.Tensor(np.random.rand(batch_size * sampling_size, 1))
        if is_recurrent:
            z, hx = model(states, hx, tau, sampling_size)
        else:
            z = model(states, tau, sampling_size)
    else:
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        if is_recurrent:
            z, hx = model(states, hx, tau, num_tau_sample)
        else:
            z = model(states, tau, num_tau_sample)
    if is_recurrent:
        return z, hx
    else:
        return z


def construct_batch_data(feature_len, dataset, batch_size, device):
    states, next_states = [], []
    dataset = list(dataset.memory)
    count = 0
    episodes_states = []
    episodes_next_states = []
    episodes_len = []
    assert len(dataset) != 0, "ReplayMemory is empty!"
    for i, data in enumerate(dataset):
        states.append(data.state.cpu().numpy().reshape(-1))
        next_states.append(data.next_state.cpu().numpy().reshape(-1))
        count += 1
        if not data.mask:
            episodes_states.append(states)
            episodes_next_states.append(next_states)
            episodes_len.append(count)
            count = 0
            states = []
            next_states = []
    max_len = max(episodes_len)
    for i, _ in enumerate(episodes_states):
        episodes_states[i] = np.concatenate((episodes_states[i], np.zeros((max_len - len(episodes_states[i]), feature_len))), axis=0)
        episodes_next_states[i] = np.concatenate((episodes_next_states[i], np.zeros((max_len - len(episodes_next_states[i]), feature_len))), axis=0)

        episodes_states[i] = torch.Tensor(episodes_states[i]).to(device)
        episodes_next_states[i] = torch.Tensor(episodes_next_states[i]).to(device)

    episodes_states = torch.stack(episodes_states).to(device)
    episodes_next_states = torch.stack(episodes_next_states).to(device)
    episodes_len = torch.Tensor(episodes_len).to(device)[:, None, None]

    tensor_dataset = torch.utils.data.TensorDataset(episodes_states, episodes_next_states, episodes_len)
    all_indices = np.arange(episodes_states.size()[0])
    np.random.shuffle(all_indices)
    train_indices = all_indices[:int(len(all_indices) * 90 / 100)]
    test_indices = all_indices[int(len(all_indices) * 90 / 100):]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dl = DataLoader(tensor_dataset, batch_size, sampler=train_sampler)
    test_dl = DataLoader(tensor_dataset, batch_size, sampler=test_sampler)
    print("Data is ready for the task!")
    return train_dl, test_dl, max_len


def construct_single_data(args, env, policy, horizon):
    when_anomaly_occurs = []
    states = []
    next_states = []
    state = env.reset()
    state = torch.Tensor(state).unsqueeze(0)
    done, ep_reward = False, 0
    counter = 0
    while not done:
        action, z_values = get_action(state, policy, -1, env, args.num_quantile_sample)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.Tensor(next_state).unsqueeze(0)
        states.append(state)
        next_states.append(next_state)
        state = next_state
        ep_reward += reward
        if counter < args.anomaly_inserted - horizon:
            when_anomaly_occurs.append(0)
        else:
            when_anomaly_occurs.append(1)
        counter += 1
    return states, next_states, when_anomaly_occurs


def learn_model(model, optimizer, memory, max_len, gru_size, num_tau_sample, device, has_memory, clip_value, feature_len):
    total_loss = 0
    count = 0
    model.train()
    for s_batch, mc_returns, _ in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                _, loss, h_memory = train_model(model, optimizer, h_memory.detach().to(device), s, mc_return,
                                                len(s_batch), num_tau_sample, device, has_memory, clip_value, feature_len)
                total_loss += loss.item()
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                _, loss = train_model(model, optimizer, None, s, mc_return, len(s_batch), num_tau_sample, device,
                                      has_memory, clip_value, feature_len)
                total_loss += loss.item()
                count += 1
    return total_loss / count


def evaluate_model(model, memory, max_len, gru_size, num_tau_sample, device, best_total_loss, path, has_memory, feature_len):
    total_loss = 0
    count = 0
    model.eval()
    for s_batch, mc_returns, _ in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                _, loss, h_memory = test_model(model, h_memory.detach().to(device), s, mc_return, len(s_batch),
                                               num_tau_sample, device, has_memory, feature_len)
                total_loss += loss
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                _, loss = test_model(model, None, s, mc_return, len(s_batch), num_tau_sample, device, has_memory,
                                     feature_len)
                total_loss += loss
                count += 1
    print("test loss :", total_loss.item() / count)
    if total_loss.item() / count <= best_total_loss:
        print("Saving the best model!")
        best_total_loss = total_loss.item() / count
        torch.save(model.state_dict(), path)
    return total_loss.item() / count, best_total_loss


def testing_model(policy, predictor, env, gru_size, num_tau_sample, device, has_memory, feature_len, num_quantile_sample):
    policy.eval()
    predictor.eval()
    estimated_dists = []
    actual_returns = []
    total_loss = 0
    with torch.no_grad():
        state = env.reset()
        state = torch.Tensor(state).unsqueeze(0)
        done, ep_reward = False, 0
        h_memory = torch.zeros(len(state) * num_tau_sample, gru_size)
        while not done:
            action, z_values = get_action(state, policy, -1, env, num_quantile_sample)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state).unsqueeze(0)
            if has_memory:
                value_return, loss, h_memory = test_model(predictor, h_memory.detach().to(device), state, next_state,
                                                          len(state), num_tau_sample, device, has_memory, feature_len)
            else:
                value_return, loss = test_model(predictor, None, state, next_state, len(state), num_tau_sample, device,
                                                has_memory, feature_len)
            estimated_dists.append(value_return.squeeze(0).squeeze(1).detach().cpu().numpy())
            actual_returns.append(next_state.cpu().numpy().squeeze(0))
            state = next_state
            ep_reward += reward
            total_loss += loss
    print("------Test score => {}".format(ep_reward))
    print("------Test loss => {}".format(total_loss.item() / len(estimated_dists)))
    return total_loss.item(), np.array(actual_returns), np.array(estimated_dists)


def ss_learn_model(model, optimizer, memory, max_len, gru_size, num_tau_sample, device, has_memory, epsilon, clip_value, feature_len):
    total_loss = 0
    count = 0
    model.train()
    s_hat = None
    for s_batch, mc_returns, _ in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, h_memory = train_model(model, optimizer, h_memory.detach().to(device), s, mc_return,
                                                        len(s_batch), num_tau_sample, device, has_memory,
                                                        clip_value, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss, h_memory = train_model(model, optimizer, h_memory.detach().to(device), s_hat.detach(),
                                                        mc_return, len(s_batch), num_tau_sample, device, has_memory,
                                                        clip_value, feature_len)
                total_loss += loss.item()
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss = train_model(model, optimizer, None, s, mc_return, len(s_batch), num_tau_sample,
                                              device, has_memory, clip_value, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss = train_model(model, optimizer, None, s_hat.detach(), mc_return, len(s_batch),
                                              num_tau_sample, device, has_memory, clip_value, feature_len)
                total_loss += loss.item()
                count += 1
    return total_loss / count


def epsilon_decay(epsilon, num_iterations, iteration, decay_type="linear", k=0.997):
    if decay_type == "linear":
        step = 1 / (num_iterations * 2)
        return round(epsilon - step, 6)
    elif decay_type == "exponential":
        return max(k ** iteration, 0.5)


def ss_evaluate_model(model, memory, max_len, gru_size, num_tau_sample, device, best_total_loss, path, has_memory, epsilon, feature_len):
    total_loss = 0
    count = 0
    model.eval()
    s_hat = None
    for s_batch, mc_returns, _ in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, h_memory = test_model(model, h_memory.detach().to(device), s, mc_return, len(s_batch),
                                                       num_tau_sample, device, has_memory, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss, h_memory = test_model(model, h_memory.detach().to(device), s_hat.detach(), mc_return,
                                                       len(s_batch), num_tau_sample, device, has_memory, feature_len)
                s_hat = s_hat.squeeze(2)
                total_loss += loss
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss = test_model(model, None, s, mc_return, len(s_batch), num_tau_sample, device,
                                             has_memory, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss = test_model(model, None, s_hat.detach(), mc_return, len(s_batch), num_tau_sample,
                                             device, has_memory, feature_len)
                s_hat = s_hat.squeeze(2)
                total_loss += loss
                count += 1
    print("test loss :", total_loss.item() / count)
    if total_loss.item() / count <= best_total_loss:
        print("Saving the best model!")
        best_total_loss = total_loss.item() / count
        torch.save(model.state_dict(), path)
    return total_loss.item() / count, best_total_loss


def plot_losses(train_loss, test_loss, results_folder, has_memory, scheduled_sampling=False):
    plt.plot(train_loss, label="training loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    path_suffix = "_ss" if scheduled_sampling else ""
    if has_memory:
        plt.savefig(os.path.join(results_folder, "rnn_autoregressive_loss" + path_suffix + ".png"))
    else:
        plt.savefig(os.path.join(results_folder, "ff_autoregressive_loss" + path_suffix + ".png"))
    plt.clf()


def plot_accuracy(feature_len, mc_returns, distributions, result_folder, anomaly_insertion, horizon, has_memory):
    fig, axs = plt.subplots(math.ceil(feature_len / 3), 3, figsize=(20, 20))
    r, c = 0, 0
    for i in range(feature_len):
        for xxx in range(len(distributions[:, i])):
            axs[r, c].scatter(np.zeros(len(distributions[:, i][xxx])) + xxx, distributions[:, i][xxx],
                              marker='.', color='teal')
        axs[r, c].plot(mc_returns[:, i][:len(distributions[:, i])], color='limegreen')
        axs[r, c].axvline(x=anomaly_insertion, color='black')
        axs[r, c].set_title("Feature: " + str(i))
        axs[r, c].set(xlabel='time', ylabel='value')
        if r < math.ceil(feature_len / 3) - 1:
            r += 1
        else:
            c += 1
            r = 0
    labels = ["Predictions", "Anomalous returns", "Anomaly injection"]
    fig.legend(labels=labels, labelcolor=['teal', 'limegreen', 'black'], handlelength=0)
    fig.suptitle("Autoregressive model predictions vs. true data\n"
                 "Horizon: " + str(horizon) + "\n"
                 + ("Model: RNN" if has_memory else "Model: FF") + "\n")
    fig.tight_layout()
    # fig.show()
    if has_memory:
        fig.savefig(os.path.join(result_folder, "rnn_predictions_vs_truedata.png"))
    else:
        fig.savefig(os.path.join(result_folder, "ff_predictions_vs_truedata.png"))
    plt.clf()
    plt.cla()
    plt.close()


def autoregressive_anomaly_detection(all_predictors, states_list, next_states_list, gru_size, num_tau_sample, device,
                                     anomaly_occurrence, feature_len, horizon, sampling_size, has_memory):
    all_value_returns = []
    for predictor in all_predictors:
        dists = []
        actual_returns = []
        anomaly_scores = []
        predictor.eval()
        h_memory = torch.zeros(len(states_list[0]) * sampling_size, gru_size)
        for i in range(len(states_list)):
            if has_memory:
                value_return, h_memory = feed_forward(predictor, h_memory.detach().to(device), states_list[i],
                                                      len(states_list[i]), num_tau_sample, sampling_size,
                                                      has_memory, tree_root=True)
                # unaffected_h_memory: a trick to keep memory of rnn unaffected
                unaffected_h_memory = h_memory
                # loop to go over the horizon
                for j in range(1, horizon):
                    tmp_h_memory = []
                    tmp_value_return = []
                    value_return_t = value_return
                    h_memory_t = h_memory
                    for sample in range(sampling_size):
                        value_return, h_memory = feed_forward(predictor, h_memory_t[sample, :].detach().reshape(1, -1).to(device),
                                                              value_return_t[:, :, sample], len(value_return_t),
                                                              num_tau_sample, sampling_size, has_memory, tree_root=False)
                        tmp_h_memory.append(h_memory)
                        tmp_value_return.append(value_return)
                    h_memory = torch.stack(tmp_h_memory).squeeze(1)
                    value_return = torch.stack(tmp_value_return).squeeze(1).reshape(1, feature_len, -1)
                h_memory = unaffected_h_memory

            else:
                value_return = feed_forward(predictor, None, states_list[i], len(states_list[i]), num_tau_sample,
                                            sampling_size, has_memory, tree_root=True)
                # loop to go over the horizon
                for j in range(1, horizon):
                    tmp_value_return = []
                    value_return_t = value_return
                    for sample in range(sampling_size):
                        value_return = feed_forward(predictor, None, value_return_t[:, :, sample], len(value_return_t),
                                                    num_tau_sample, sampling_size, has_memory, tree_root=False)
                        tmp_value_return.append(value_return)
                    value_return = torch.stack(tmp_value_return).squeeze(1).reshape(1, feature_len, -1)

            dists.append(value_return.squeeze(0).detach().cpu().numpy())
            actual_returns.append(next_states_list[i].squeeze(0).detach().cpu().numpy())
        all_value_returns.append(dists)
    all_value_returns = np.concatenate(np.array(all_value_returns), axis=2)
    for i in range(len(states_list)):
        anomaly_scores.append(measure_as(all_value_returns[i], next_states_list[i].squeeze(0).cpu().numpy(), feature_len))

    separated_results = separated_confusion_matrix(anomaly_scores, anomaly_occurrence, feature_len)
    averaged_as = np.array(anomaly_scores).mean(axis=1)
    maxed_as = np.array(anomaly_scores).max(axis=1)
    merged_avg_auc, avg_fa_rate, fpr, tpr = merged_confusion_matrix(averaged_as, anomaly_occurrence)
    merged_max_auc, max_fa_rate, _, _ = merged_confusion_matrix(maxed_as, anomaly_occurrence)
    # print("Averaged AUC:", merged_avg_auc)
    # print("Max AUC:", merged_max_auc)
    return separated_results, np.array(actual_returns), np.array(dists), merged_avg_auc, merged_max_auc, \
           anomaly_scores, avg_fa_rate, max_fa_rate, fpr, tpr


def measure_as(distribution, actual_return, feature_len):
    anomaly_scores = []
    for i in range(feature_len):
        anomaly_scores.append(k_nearest_neighbors(distribution[i, :], actual_return[i]))
    return np.array(anomaly_scores)


def k_nearest_neighbors(distribution, actual_return):
    neigh = NearestNeighbors(n_neighbors=distribution.shape[0])
    neigh.fit(distribution.reshape(-1, 1))
    distances, indices = neigh.kneighbors(np.array(actual_return).reshape(-1, 1))
    return distances.mean()


def separated_confusion_matrix(scores, anom_occurrence, feature_len):
    results = {}
    for i in range(feature_len):
        fpr, tpr, thresholds = roc_curve(anom_occurrence, np.array(scores)[:, i])
        auc = sklearn.metrics.auc(fpr, tpr)
        results[i] = (fpr, tpr, thresholds, auc)
    return results


def false_alarm_rater(thresholds, scores, nominal_len):
    fa_rates = []
    for th in thresholds:
        no_false_alarms = len(scores[:nominal_len][scores[:nominal_len] > th])
        fa_rates.append(no_false_alarms / nominal_len)
    return np.array(fa_rates).mean()


def merged_confusion_matrix(scores, anom_occurrence):
    fpr, tpr, thresholds = roc_curve(anom_occurrence, scores)
    auc = sklearn.metrics.auc(fpr, tpr)
    nominal_len = anom_occurrence.index(1)
    far = false_alarm_rater(thresholds, scores, nominal_len)
    return auc, far, fpr, tpr


def bootstrap_cusum(anomaly_scores, feature_len=18):
    cusums = {}
    for key in range(feature_len):
        cusums[key] = []
        cusums[key].append(0)
        as_mean = anomaly_scores[:, key].mean()
        for i in range(1, len(anomaly_scores[:, key])):
            cusums[key].append(cusums[key][i - 1] + anomaly_scores[:, key][i - 1] - as_mean)
    return cusums


def original_cusum(anomaly_scores, feature_len=18):
    cusums = {}
    for key in range(feature_len):
        cusums[key] = []
        cusums[key] = detect_cusum(anomaly_scores[:, key], threshold=0.01, drift=.0018, ending=True, show=False)[0]
    return cusums


def load_predictive_models(args, input_output_len, device):
    if args.is_recurrent:
        model = AutoregressiveRecurrentIQN(input_output_len, args.gru_units, args.quantile_embedding_dim,
                                           args.num_quantile_sample, device)
    elif args.is_recurrent_v2:
        model = AutoregressiveRecurrentIQN_v2(input_output_len, args.gru_units, args.quantile_embedding_dim,
                                              args.num_quantile_sample, device)
    else:
        model = AutoregressiveIQN(input_output_len, args.quantile_embedding_dim,
                                  args.num_quantile_sample, device)
    return model


def input_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictive_model_training', action='store_true', default=False,
                        help="To train autoregressive models")
    parser.add_argument('--predictive_model_testing', action='store_true', default=False,
                        help="To test autoregressive models")
    parser.add_argument('--anomaly_detection', action='store_true', default=False,
                        help="Do the AD when anomalies injected into the system")
    parser.add_argument('--horizon_comparison_as', action='store_true', default=False,
                        help="Studying the affect of horizon on anomaly scores and AUCs")
    parser.add_argument('--samplesize_comparison_as', action='store_true', default=False,
                        help="Studying the affect of sampling size on anomaly scores and AUCs")
    parser.add_argument('--avgvsmax_comparison_as', action='store_true', default=False,
                        help="Studying the affect of combining anomaly scores using avg vs. max on AUCs")
    parser.add_argument('--dataset_analysis', action='store_true', default=False,
                        help="Analyzing dataset")
    parser.add_argument('--dists_cdf', action='store_true', default=False,
                        help="Studying CDFs of internal distributions")
    parser.add_argument('--detection_delay', action='store_true', default=False,
                        help="Measuring the delay in detecting anomalies")
    parser.add_argument('--is_recurrent', action='store_true', default=False,
                        help="Determines whether the model has memory or not")
    parser.add_argument('--is_recurrent_v2', action='store_true', default=False,
                        help="Determines whether the model has memory or not -- v2 RNN model")
    parser.add_argument('--feature_part_analysis', action='store_true', default=False,
                        help="Analyzing feature participation is calculating anomaly scores")
    parser.add_argument('--scheduled_sampling_training', action='store_true', default=False,
                        help="To train autoregressive models using scheduled sampling")
    parser.add_argument('--predictive_model_paths', nargs='+', type=str,
                        help="Path to all predictive models")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--gru_units', type=int, default=64,
                        help="Number of cells in the GRU")
    parser.add_argument('--num_quantile_sample', type=int, default=64,
                        help="Number of quantile samples for IQN")
    parser.add_argument('--policy_num_quantile_sample', type=int, default=32,
                        help="Number of quantile samples for policy IQN")
    parser.add_argument('--num_tau_sample', type=int, default=1,
                        help="Number of tau samples for IQN, sets the distribution sampling size.")
    parser.add_argument('--quantile_embedding_dim', type=int, default=128,
                        help="Qunatiles embedding dimension in IQN")
    parser.add_argument('--policy_quantile_embedding_dim', type=int, default=64,
                        help="Qunatiles embedding dimension in policy IQN")
    parser.add_argument('--test_interval', type=int, default=10,
                        help="Intervals between train and test")
    parser.add_argument('--num_iterations', type=int, default=3000,
                        help="Number of iterations to update model")
    parser.add_argument('--env_name', type=str,
                        help="Name of the main environment: to train, test, update models, find threshold, and "
                             "calculate performance on normal envs")
    parser.add_argument('--data_path', type=str,
                        help="path to the dataset json file")
    parser.add_argument('--test_data_path', type=str,
                        help="path to the test dataset json file")
    parser.add_argument('--noisy_data_path', type=str,
                        help="path to the test dataset json file")
    parser.add_argument('--anomaly_inserted', type=int,
                        help="Time when the anomaly is inserted into the system")
    parser.add_argument('--clip_value', type=int, default=None,
                        help="Clipping gradients")
    parser.add_argument('--horizons', nargs='+', type=int,
                        help="Horizon to go forward in time")
    parser.add_argument('--sampling_sizes', nargs='+', type=int,
                        help="Size of the sampling to build the tree of distributions at time t")
    parser.add_argument('--given_fpr', type=float,
                        help='Acceptable FPR rate to calculate the threshold for anomaly detection delay')
    parser.add_argument('--decay_type', type=str, choices=["linear", "exponential"], default="linear",
                        help="How to decay epsilon in Scheduled sampling")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = input_arg_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "./models/"
    # results_path = os.path.join(base_path, args.env_name)
    results_path = os.path.join(base_path, "Acrobot-v1")
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    env = gym.make(args.env_name)
    num_features = env.observation_space.shape[0]
    policy_model_path = os.path.join(results_path, "policy.pt")
    policy_model = IQN(num_features, env.action_space.n, args.policy_quantile_embedding_dim,
                                  args.policy_num_quantile_sample, device, args.env_name)
    policy_model.load_state_dict(torch.load(policy_model_path, map_location=device))

    predictive_model = load_predictive_models(args, num_features, device)
    if args.predictive_model_training:
        print("Loading GVD training data!")
        optimal_p_memory_path = os.path.join(results_path, "optimal_p_memory.pt")
        optimal_memory = torch.load(optimal_p_memory_path, map_location=device)
        print("GVD data loaded!")
        train_rb, test_rb, max_len = construct_batch_data(num_features, optimal_memory, args.batch_size, device)
        if os.path.exists(args.predictive_model_paths[0]):
            print("Loading pre-trained model!")
            predictive_model.load_state_dict(torch.load(args.predictive_model_paths[0], map_location=device))
            print("Pre-trained model loaded:", args.predictive_model_paths[0])
        optimizer = torch.optim.Adam(predictive_model.parameters(), lr=args.lr)
        predictive_model.to(device)
        predictive_model.train()

        all_train_losses, all_test_losses = [], []
        best_total_loss = float("inf")
        for i in range(args.num_iterations):
            total_loss = learn_model(predictive_model, optimizer, train_rb, max_len, args.gru_units, args.num_tau_sample,
                                     device, args.is_recurrent or args.is_recurrent_v2, args.clip_value, num_features)
            if i % args.test_interval == 0:
                print("train loss : {}".format(total_loss))
                all_train_losses.append(total_loss)
                avg_eval_loss, best_total_loss = evaluate_model(predictive_model, test_rb, max_len, args.gru_units,
                                                                args.num_tau_sample, device, best_total_loss,
                                                                args.predictive_model_paths[0],
                                                                args.is_recurrent or args.is_recurrent_v2, num_features)
                all_test_losses.append(avg_eval_loss)
                plot_losses(all_train_losses, all_test_losses, results_path, args.is_recurrent or args.is_recurrent_v2)
        final_model_path = args.predictive_model_paths[0].replace(".pt", "_final.pt")
        print("Saving the last model!")
        torch.save(predictive_model.state_dict(), final_model_path)

    elif args.predictive_model_testing:
        print("Loading GVD training data!")
        optimal_p_memory_path = os.path.join(results_path, "optimal_p_memory.pt")
        optimal_memory = torch.load(optimal_p_memory_path, map_location=device)
        print("GVD data loaded!")
        predictive_model.load_state_dict(torch.load(args.predictive_model_paths[0], map_location=device))
        print("Trained model loaded:", args.predictive_model_paths[0])
        predictive_model.to(device)
        predictive_model.eval()

        item_loss, actual_returns, dist_returns = testing_model(policy_model, predictive_model, env, args.gru_units,
                                                                args.num_tau_sample, device,
                                                                args.is_recurrent or args.is_recurrent_v2, num_features,
                                                                args.num_quantile_sample)
        plot_accuracy(num_features, actual_returns, dist_returns, results_path, args.anomaly_inserted, args.horizons[0],
                      args.is_recurrent or args.is_recurrent_v2)

    elif args.anomaly_detection:
        env.when_anomaly_starts = args.anomaly_inserted
        all_predictive_models = []
        for model_path in args.predictive_model_paths:
            predictive_model = load_predictive_models(args, num_features, device)
            predictive_model.load_state_dict(torch.load(model_path, map_location=device))
            predictive_model.to(device)
            predictive_model.eval()
            all_predictive_models.append(predictive_model)
            print("Trained model loaded:", model_path)
        fprs, tprs = [], []
        for h in args.horizons:
            for ss in args.sampling_sizes:

                all_avg_aucs = []
                all_max_aucs = []
                all_seperated_aucs = []
                all_avg_false_alarm_rates = []
                all_max_false_alarm_rates = []
                on_features_bootstrap_cusums = []
                on_scores_bootstrap_cusums = []
                on_features_original_cusums = []
                on_scores_original_cusums = []
                with torch.no_grad():
                    for ep in range(args.num_iterations):
                        states, next_states, when_anomaly_occurs = construct_single_data(args, env, policy_model, h)
                        seperated_results, noisy_acs, dists, merged_avg_auc, \
                        merged_max_auc, ass, avg_f_a_rate, max_f_a_rate, fpr, tpr = autoregressive_anomaly_detection(
                                                                            all_predictive_models, states, next_states,
                                                                            args.gru_units, args.num_tau_sample, device,
                                                                            when_anomaly_occurs, num_features, h, ss,
                                                                            args.is_recurrent or args.is_recurrent_v2)
                        fprs.append(fpr)
                        tprs.append(tpr)
                        on_features_cusum_changepoints = bootstrap_cusum(noisy_acs, num_features)
                        on_scores_cusum_changepoints = bootstrap_cusum(np.array(ass), num_features)
                        on_features_bootstrap_cusums.append(on_features_cusum_changepoints[0])
                        on_scores_bootstrap_cusums.append(on_scores_cusum_changepoints[0])
                        on_features_cusum_changepoints = original_cusum(noisy_acs, num_features)
                        on_scores_cusum_changepoints = original_cusum(np.array(ass), num_features)
                        on_features_original_cusums.append(on_features_cusum_changepoints[0])
                        on_scores_original_cusums.append(on_scores_cusum_changepoints[0])

                        all_avg_aucs.append(merged_avg_auc)
                        all_max_aucs.append(merged_max_auc)
                        all_avg_false_alarm_rates.append(avg_f_a_rate)
                        all_max_false_alarm_rates.append(max_f_a_rate)
                        all_seperated_aucs.append(np.array([item[3] for item in list(seperated_results.values())]))

                    # plot_accuracy(num_features, noisy_acs, dists, results_path, args.anomaly_inserted, args.horizons[0],
                    #               args.is_recurrent or args.is_recurrent_v2)

                warmup = 0
                for seq_i, seq in enumerate(on_features_bootstrap_cusums):
                    changes = []
                    for i in range(warmup, len(seq)):
                        if seq[i] > seq[i - 1]:
                            changes.append(i)
                            break
                    for i in range(warmup, len(seq)):
                        if seq[i] < seq[i - 1]:
                            changes.append(i)
                            break
                    change_points = [x for x in sorted(changes) if x >= 45]
                for seq_i, seq in enumerate(on_scores_bootstrap_cusums):
                    changes = []
                    for i in range(warmup, len(seq)):
                        if seq[i] > seq[i - 1]:
                            changes.append(i)
                            break
                    for i in range(warmup, len(seq)):
                        if seq[i] < seq[i - 1]:
                            changes.append(i)
                            break
                    change_points = [x for x in sorted(changes) if x >= 45]
                on_features_original_changes = []
                for item in on_features_original_cusums:
                    change_points = [x[1] for x in enumerate(item) if x[1] > warmup]
                    if len(change_points) > 0:
                        on_features_original_changes.append(change_points[0])
                on_scores_original_changes = []
                for item in on_scores_original_cusums:
                    change_points = [x[1] for x in enumerate(item) if x[1] > warmup]
                    if len(change_points) > 0:
                        on_scores_original_changes.append(change_points[0])
                print("********************************* H, SS:", h, ss)
                if len(all_avg_aucs) != 0:
                    print("Averaging all avg AUCs:", round(sum(all_avg_aucs) / len(all_avg_aucs), 2))
                if len(all_max_aucs) != 0:
                    print("Averaging all max AUCs:", round(sum(all_max_aucs) / len(all_max_aucs), 2))
                print("Average of change-point detection times - using features and original CUSUM:",
                      np.array(on_features_original_changes).mean())
                print("Average of change-point detection times - using anomaly scores and original CUSUM:",
                      np.array(on_scores_original_changes).mean())
                print("False alarm rate - using average scores:", round(np.array(all_avg_false_alarm_rates).mean(), 2))
                print("False alarm rate - using max scores:", round(np.array(all_max_false_alarm_rates).mean(), 2))
    elif args.scheduled_sampling_training:
        predictive_model_path = args.predictive_model_paths[0].replace(".pt", "_ss.pt")
        final_predictive_model_path = predictive_model_path.replace("_ss.pt", "_ss_final.pt")
        print("Loading GVD training data!")
        optimal_p_memory_path = os.path.join(base_path, args.env_name, "optimal_p_memory.pt")
        optimal_memory = torch.load(optimal_p_memory_path, map_location=device)
        print("GVD data loaded!")
        train_rb, test_rb, max_len = construct_batch_data(num_features, optimal_memory, args.batch_size, device)
        if os.path.exists(predictive_model_path):
            print("Loading pre-trained model!")
            predictive_model.load_state_dict(torch.load(predictive_model_path, map_location=device))
            print("Trained model loaded:", predictive_model_path)
        optimizer = torch.optim.Adam(predictive_model.parameters(), lr=args.lr)
        predictive_model.to(device)
        predictive_model.train()
        epsilon = 1
        all_train_losses, all_test_losses = [], []
        best_total_loss = float("inf")
        for i in range(args.num_iterations):
            print("----------------------> EPSILON:", epsilon)
            total_loss = ss_learn_model(predictive_model, optimizer, train_rb, max_len, args.gru_units,
                                        args.num_tau_sample, device, args.is_recurrent or args.is_recurrent_v2, epsilon,
                                        args.clip_value, num_features)
            if i % args.test_interval == 0:
                print("train loss : {}".format(total_loss))
                all_train_losses.append(total_loss)
                avg_eval_loss, best_total_loss = ss_evaluate_model(predictive_model, test_rb, max_len, args.gru_units,
                                                                   args.num_tau_sample, device, best_total_loss,
                                                                   predictive_model_path,
                                                                   args.is_recurrent or args.is_recurrent_v2, epsilon,
                                                                   num_features)
                all_test_losses.append(avg_eval_loss)
                plot_losses(all_train_losses, all_test_losses, results_path, args.is_recurrent or args.is_recurrent_v2,
                            scheduled_sampling=True)
            epsilon = epsilon_decay(epsilon, args.num_iterations, i, args.decay_type)
        print("Saving the last model!")
        torch.save(predictive_model.state_dict(), final_predictive_model_path)
