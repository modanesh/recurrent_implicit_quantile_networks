import os
import gym
import math
import torch
import random
import sklearn
import argparse
import numpy as np
import pybullet_envs
import torch.nn as nn
import seaborn as sns
import env_preparation
import matplotlib.pyplot as plt
from detecta import detect_cusum
from stable_baselines3 import TD3
from sklearn.metrics import roc_curve
import torch.nn.functional as functional
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, DummyVecEnv
sns.set()


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
        x = ghx + functional.relu(self.fc_2(ghx))
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


def construct_batch_data(feature_len, dataset, device):
    episodes_states = []
    episodes_next_states = []
    episodes_len = []
    for i, episode in enumerate(dataset):
        episodes_len.append(len(episode))
    max_len = max(episodes_len) - 1
    for i, episode in enumerate(dataset):
        # get rid of features added by TimeFeatureWrapper
        episode = np.array(episode).squeeze(1)[:,:-1]
        episodes_states.append(torch.Tensor(
            np.concatenate((episode[:-1, :], np.zeros((max_len - len(episode[:-1, :]), feature_len))), axis=0)))
        episodes_next_states.append(torch.Tensor(
            np.concatenate((episode[1:, :], np.zeros((max_len - len(episode[1:, :]), feature_len))), axis=0)))

    episodes_states = torch.stack(episodes_states).to(device)
    episodes_next_states = torch.stack(episodes_next_states).to(device)

    tensor_dataset = torch.utils.data.TensorDataset(episodes_states, episodes_next_states)
    return tensor_dataset


def data_splitting(tensor_dataset, batch_size, features_min, features_max, device):
    # prevent division by zero in normalization
    no_need_normalization = np.where((features_min == features_max))[0]
    normalized_states = (tensor_dataset[0][0].cpu().numpy() - features_min) / (features_max - features_min)
    normalized_n_states = (tensor_dataset[0][1].cpu().numpy() - features_min) / (features_max - features_min)
    for index in no_need_normalization:
        normalized_states[:, index] = features_min[index]
        normalized_n_states[:, index] = features_min[index]
    normalized_tensor_dataset = torch.utils.data.TensorDataset(torch.Tensor(normalized_states).to(device),
                                                               torch.Tensor(normalized_n_states).to(device))
    all_indices = np.arange(len(normalized_tensor_dataset))
    max_len = len(normalized_tensor_dataset[0][0])
    np.random.shuffle(all_indices)
    train_indices = all_indices[:int(len(all_indices) * 90 / 100)]
    test_indices = all_indices[int(len(all_indices) * 90 / 100):]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dl = DataLoader(normalized_tensor_dataset, batch_size, sampler=train_sampler)
    test_dl = DataLoader(normalized_tensor_dataset, batch_size, sampler=test_sampler)
    return train_dl, test_dl, max_len


def train_model(model, optimizer, hx, states, target, batch_size, num_tau_sample, device, clip_value, feature_len):
    tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
    states = states.reshape(states.shape[0], 1, -1)
    if hx is not None:
        z, hx = model(states, hx, tau, num_tau_sample)
    else:
        z = model(states, tau, num_tau_sample)
    T_z = target.reshape(target.shape[0], 1, -1).expand(-1, num_tau_sample, feature_len).transpose(1, 2)

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
    return z.squeeze(2), loss, hx


def test_model(model, hx, states, target, batch_size, num_tau_sample, device, feature_len):
    tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
    states = states.reshape(states.shape[0], 1, -1)
    if hx is not None:
        z, hx = model(states, hx, tau, num_tau_sample)
    else:
        z = model(states, tau, num_tau_sample)
    T_z = target.reshape(target.shape[0], 1, -1).expand(-1, num_tau_sample, feature_len).transpose(1, 2)

    error_loss = T_z - z
    huber_loss = functional.smooth_l1_loss(z, T_z.detach(), reduction='none')
    if num_tau_sample == 1:
        tau = torch.arange(0, 1, 1 / 100).view(1, 100)
    else:
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

    loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
    loss = loss.mean()
    return z, loss, hx


def feed_forward(model, hx, states, batch_size, num_tau_sample, sampling_size, tree_root=False):
    states = states.reshape(states.shape[0], 1, -1)
    if tree_root:
        tau = torch.Tensor(np.random.rand(batch_size * sampling_size, 1))
        if hx is not None:
            z, hx = model(states, hx, tau, sampling_size)
        else:
            z = model(states, tau, sampling_size)
    else:
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        if hx is not None:
            z, hx = model(states, hx, tau, num_tau_sample)
        else:
            z = model(states, tau, num_tau_sample)
    return z, hx


def ss_learn_model(model, optimizer, memory, max_len, gru_size, num_tau_sample, device, epsilon, clip_value, feature_len,
                   has_memory):
    total_loss = 0
    count = 0
    model.train()
    s_hat = None
    for s_batch, mc_returns in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, h_memory = train_model(model, optimizer, h_memory.detach().to(device), s, mc_return,
                                                        len(s_batch), num_tau_sample, device, clip_value, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss, h_memory = train_model(model, optimizer, h_memory.detach().to(device), s_hat.detach(),
                                                        mc_return, len(s_batch), num_tau_sample, device, clip_value,
                                                        feature_len)
                total_loss += loss.item()
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, _ = train_model(model, optimizer, None, s, mc_return, len(s_batch), num_tau_sample,
                                                 device, clip_value, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss, _ = train_model(model, optimizer, None, s_hat.detach(), mc_return, len(s_batch),
                                                 num_tau_sample, device, clip_value, feature_len)
                total_loss += loss.item()
                count += 1
    return total_loss / count


def ss_evaluate_model(model, memory, max_len, gru_size, num_tau_sample, device, best_total_loss, path, epsilon, feature_len, has_memory):
    total_loss = 0
    count = 0
    model.eval()
    s_hat = None
    for s_batch, mc_returns in memory:
        if has_memory:
            h_memory = None
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if h_memory is None:
                    h_memory = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, h_memory = test_model(model, h_memory.detach().to(device), s, mc_return, len(s_batch),
                                                       num_tau_sample, device, feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss, h_memory = test_model(model, h_memory.detach().to(device), s_hat.detach(), mc_return,
                                                       len(s_batch), num_tau_sample, device, feature_len)
                s_hat = s_hat.squeeze(2)
                total_loss += loss
                count += 1
        else:
            for i in range(max_len):
                s, mc_return = s_batch[:, i], mc_returns[:, i]
                if random.random() <= epsilon or s_hat is None:
                    s_hat, loss, _ = test_model(model, None, s, mc_return, len(s_batch), num_tau_sample, device,
                                                feature_len)
                else:
                    if len(s_hat) != len(s):
                        s_hat = s_hat[:len(s)]
                    s_hat, loss, _ = test_model(model, None, s_hat.detach(), mc_return, len(s_batch), num_tau_sample,
                                                device, feature_len)
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


def epsilon_decay(epsilon, num_iterations, iteration, decay_type="linear", k=0.997):
    if decay_type == "linear":
        step = 1 / (num_iterations * 2)
        return round(epsilon - step, 6)
    elif decay_type == "exponential":
        return max(k ** iteration, 0.5)


def measure_as(distribution, actual_return, input_len):
    anomaly_scores = []
    for i in range(input_len):
        anomaly_scores.append(k_nearest_neighbors(distribution[i, :], actual_return[i]))
    return np.array(anomaly_scores)


def k_nearest_neighbors(distribution, actual_return):
    neigh = NearestNeighbors(n_neighbors=distribution.shape[0])
    neigh.fit(distribution.reshape(-1, 1))
    distances, indices = neigh.kneighbors(np.array(actual_return).reshape(-1, 1))
    return distances.mean()


def separated_confusion_matrix(scores, anom_occurrence, input_len):
    results = {}
    for i in range(input_len):
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
    nominal_len = np.where(anom_occurrence==1)[0][0]
    far = false_alarm_rater(thresholds, scores, nominal_len)
    return auc, far


def ar_anomaly_detection(predictor, gru_size, num_tau_sample, sampling_size, device, feature_len, episode, horizon,
                         anomaly_occurrence, has_memory):
    predictor.eval()
    estimated_dists = []
    anomaly_scores = []
    if has_memory:
        h_memory = torch.zeros(len(episode[0]) * sampling_size, gru_size)
        for i in range(len(episode) - horizon):
            state = episode[i][:, :feature_len]
            state = torch.Tensor(state)
            value_return, h_memory = feed_forward(predictor, h_memory.detach().to(device), state, len(state),
                                                  num_tau_sample, sampling_size, tree_root=True)
            # unaffected_h_memory: a trick to keep memory of rnn unaffected
            unaffected_h_memory = h_memory
            for j in range(1, horizon):
                tmp_h_memory = []
                tmp_value_return = []
                value_return_t = value_return
                h_memory_t = h_memory
                for sample in range(sampling_size):
                    value_return, h_memory = feed_forward(predictor, h_memory_t[sample, :].detach().reshape(1, -1),
                                                          value_return_t[:, :, sample], len(value_return_t),
                                                          num_tau_sample, sampling_size, tree_root=False)
                    tmp_h_memory.append(h_memory)
                    tmp_value_return.append(value_return)
                h_memory = torch.stack(tmp_h_memory).squeeze(1)
                value_return = torch.stack(tmp_value_return).squeeze(1).reshape(1, feature_len, -1)

            h_memory = unaffected_h_memory
            estimated_dists.append(value_return.squeeze(0).detach().cpu().numpy())
            anomaly_score = measure_as(value_return.squeeze(0).detach().cpu().numpy(),
                                       episode[i + horizon][:, :feature_len].squeeze(0), feature_len)
            anomaly_scores.append(anomaly_score)
    else:
        for i in range(len(episode) - horizon):
            state = episode[i][:, :feature_len]
            state = torch.Tensor(state)
            value_return, _ = feed_forward(predictor, None, state, len(state), num_tau_sample, sampling_size,
                                           tree_root=True)
            for j in range(1, horizon):
                tmp_value_return = []
                value_return_t = value_return
                for sample in range(sampling_size):
                    value_return, _ = feed_forward(predictor, None, value_return_t[:, :, sample], len(value_return_t),
                                                   num_tau_sample, sampling_size, tree_root=False)
                    tmp_value_return.append(value_return)
                value_return = torch.stack(tmp_value_return).squeeze(1).reshape(1, feature_len, -1)

            estimated_dists.append(value_return.squeeze(0).detach().cpu().numpy())
            anomaly_score = measure_as(value_return.squeeze(0).detach().cpu().numpy(),
                                       episode[i + horizon][:, :feature_len].squeeze(0), feature_len)
            anomaly_scores.append(anomaly_score)
    separated_results = separated_confusion_matrix(anomaly_scores, anomaly_occurrence, feature_len)
    averaged_as = np.array(anomaly_scores).mean(axis=1)
    maxed_as = np.array(anomaly_scores).max(axis=1)
    merged_avg_auc, avg_fa_rate = merged_confusion_matrix(averaged_as, anomaly_occurrence)
    merged_max_auc, max_fa_rate = merged_confusion_matrix(maxed_as, anomaly_occurrence)
    # print("Averaged AUC:", merged_avg_auc)
    # print("Max AUC:", merged_max_auc)
    return separated_results, np.array(episode).squeeze(1), np.array(estimated_dists), merged_avg_auc, merged_max_auc, \
           anomaly_scores, avg_fa_rate, max_fa_rate


def plot_accuracy(feature_len, mc_returns, h_distributions, result_folder, anomaly_insertion, horizons, env_name):
    fig, axs = plt.subplots(math.ceil(feature_len / 3), 3, figsize=(20, 20))
    colors = ['deepskyblue', 'chartreuse', 'violet']
    labels = ["True returns", "Anomaly injection"]
    used_colors = ['black', 'red']
    for h_i, h in enumerate(horizons):
        r, c = 0, 0
        for i in range(feature_len):
            for xxx in range(len(h_distributions[h][:, i])):
                axs[r, c].scatter(np.zeros(len(h_distributions[h][:, i][xxx])) + xxx + h, h_distributions[h][:, i][xxx],
                                  marker='.', color=colors[h_i])
            axs[r, c].plot(mc_returns[:, i], color='black')
            axs[r, c].axvline(x=anomaly_insertion, color='red')
            axs[r, c].set_title("Feature: " + str(i))
            axs[r, c].set(xlabel='time', ylabel='value')
            if r < math.ceil(feature_len / 3) - 1:
                r += 1
            else:
                c += 1
                r = 0
        labels.append("H="+str(h))
        used_colors.append(colors[h_i])

    fig.legend(labels=labels, labelcolor=used_colors, handlelength=0)
    fig.suptitle("Autoregressive model predictions vs. true data\n"
                 "Horizon: " + str(horizons) + "\n"
                 "Env: " + env_name + "\n")
    fig.tight_layout()
    fig.savefig(os.path.join(result_folder, "rnn_predictions_vs_truedata_h" + str(horizons) + ".png"))
    fig.show()
    plt.clf()
    plt.cla()
    plt.close()


def original_cusum(anomaly_scores, feature_len=18):
    cusums = {}
    for key in range(feature_len):
        cusums[key] = []
        cusums[key] = detect_cusum(anomaly_scores[:, key], threshold=0.01, drift=.0018, ending=True, show=False)[0]
    return cusums


def states_min_max_finder(train_dataset):
    features_min = train_dataset[0][0].min(axis=0).values.cpu().numpy()
    features_max = train_dataset[0][0].max(axis=0).values.cpu().numpy()
    return features_min, features_max


def input_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_policy', action='store_true', default=False,
                        help="To test the policy")
    parser.add_argument('--train_predictive_model', action='store_true', default=False,
                        help="To train the predictive models")
    parser.add_argument('--env_name', type=str,
                        help="Environment of interest")
    parser.add_argument('--iterations', type=int, default=1000000,
                        help="Training iterations")
    parser.add_argument('--clip_obs', type=float, default=10.,
                        help="Clipping observations for normalization")
    parser.add_argument('--power', type=float,
                        help="Power applied to the taken actions (as the nominal power). Being used for model path")
    parser.add_argument('--anomalous_power', type=float, default=None,
                        help="Power applied to the taken actions (as the anomalous power)")
    parser.add_argument('--anomaly_injection', type=int, default=None,
                        help="When to inject anomaly")
    parser.add_argument('--horizons', nargs='+', type=int,
                        help="Horizon to go forward in time")
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size")
    parser.add_argument('--quantile_embedding_dim', type=int, default=128,
                        help="Quantiles embedding dimension in IQN")
    parser.add_argument('--gru_units', type=int, default=64,
                        help="Number of cells in the GRU")
    parser.add_argument('--num_quantile_sample', type=int, default=64,
                        help="Number of quantile samples for IQN")
    parser.add_argument('--decay_type', type=str, choices=["linear", "exponential"], default="linear",
                        help="How to decay epsilon in Scheduled sampling")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--clip_value', type=int, default=None,
                        help="Clipping gradients")
    parser.add_argument('--num_tau_sample', type=int, default=1,
                        help="Number of tau samples for IQN, sets the distribution sampling size.")
    parser.add_argument('--test_interval', type=int, default=10,
                        help="Intervals between train and test")
    parser.add_argument('--anomaly_detection', action='store_true', default=False,
                        help="Do the AD when anomalies injected into the system")
    parser.add_argument('--sampling_sizes', nargs='+', type=int,
                        help="Size of the sampling to build the tree of distributions at time t")
    parser.add_argument('--case', type=int,
                        help="Which case of environment to run. Works like -v suffix in standard environment naming.")
    parser.add_argument('--is_recurrent_v2', action='store_true', default=False,
                        help="Determines whether the model has memory or not -- v2 RNN model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    powers_dict = {"Ant": 2.5, "HalfCheetah": 0.9, "Hopper": 0.75, "Walker2D": 0.4}
    anomalous_powers_dict = {"Ant": 1.5, "HalfCheetah": 0.6, "Hopper": 0.65, "Walker2D": 0.35}

    args = input_arg_parser()
    env_dir = os.path.join("./models", args.env_name)
    if not os.path.exists(env_dir):
        os.mkdir(env_dir)
    optimal_memory_path = os.path.join(env_dir, "optimal_memory.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.is_recurrent_v2:
        predictive_model_path = os.path.join(env_dir, "rnn_autoregressive_v2_ss.pt")
    else:
        predictive_model_path = os.path.join(env_dir, "ff_autoregressive_v2_ss.pt")
    policy_model_path = os.path.join(env_dir, "best_model")
    num_features = gym.make(args.env_name).observation_space.shape[0]

    if args.test_policy:
        # Load the agent
        model = TD3.load(policy_model_path)
        random_seed = random.randint(0, 1000)
        env = DummyVecEnv(
            [env_preparation.make_env(args.env_name, 0, random_seed, wrapper_class=env_preparation.TimeFeatureWrapper,
                                      env_kwargs={'power': args.anomalous_power,
                                                  'anomaly_injection': args.anomaly_injection,
                                                  'case': args.case})])
        env = VecVideoRecorder(env, env_dir, record_video_trigger=lambda x: x == 0, video_length=1000,
                               name_prefix="ap" + str(args.anomalous_power) + "_ai" + str(args.anomaly_injection))

        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=args.clip_obs)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False

        mean_reward, std_reward, observations = evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes)
        tensor_observations = construct_batch_data(num_features, observations, device)
        torch.save(tensor_observations, optimal_memory_path)
        print(f"Best model mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    elif args.train_predictive_model:
        print("Loading predictor's training data!")
        memory_rb = torch.load(optimal_memory_path, map_location=device)

        # normalizing data
        states_min, states_max = states_min_max_finder(memory_rb)

        train_rb, test_rb, max_len = data_splitting(memory_rb, args.batch_size, states_min, states_max, device)
        print("Predictor's data loaded!")
        if args.is_recurrent_v2:
            predictive_model = AutoregressiveRecurrentIQN_v2(num_features, args.gru_units, args.quantile_embedding_dim,
                                                         args.num_quantile_sample, device)
        else:
            predictive_model = AutoregressiveIQN(num_features, args.quantile_embedding_dim, args.num_quantile_sample,
                                                 device)
        if os.path.exists(predictive_model_path):
            print("Loading pre-trained model!")
            predictive_model.load_state_dict(torch.load(predictive_model_path, map_location=device))
            print("Pre-trained model loaded:", predictive_model_path)
        optimizer = torch.optim.Adam(predictive_model.parameters(), lr=args.lr)
        predictive_model.to(device)
        predictive_model.train()
        epsilon = 1
        all_train_losses, all_test_losses = [], []
        best_total_loss = float("inf")
        for i in range(args.iterations):
            total_loss = ss_learn_model(predictive_model, optimizer, train_rb, max_len, args.gru_units,
                                        args.num_tau_sample, device, epsilon, args.clip_value, num_features,
                                        args.is_recurrent_v2)
            if i % args.test_interval == 0:
                print("train loss : {}".format(total_loss))
                all_train_losses.append(total_loss)
                avg_eval_loss, best_total_loss = ss_evaluate_model(predictive_model, test_rb, max_len, args.gru_units,
                                                                   args.num_tau_sample, device, best_total_loss,
                                                                   predictive_model_path, epsilon, num_features,
                                                                   args.is_recurrent_v2)
                all_test_losses.append(avg_eval_loss)
                plot_losses(all_train_losses, all_test_losses, env_dir, args.is_recurrent_v2, scheduled_sampling=True)
            epsilon = epsilon_decay(epsilon, args.iterations, i, args.decay_type)
        final_model_path = predictive_model_path.replace(".pt", "_final.pt")
        print("Saving the last model!")
        torch.save(predictive_model.state_dict(), final_model_path)

    elif args.anomaly_detection:
        if args.is_recurrent_v2:
            predictive_model = AutoregressiveRecurrentIQN_v2(num_features, args.gru_units, args.quantile_embedding_dim,
                                                             args.num_quantile_sample, device)
        else:
            predictive_model = AutoregressiveIQN(num_features, args.quantile_embedding_dim, args.num_quantile_sample,
                                                 device)
        predictive_model.load_state_dict(torch.load(predictive_model_path, map_location=device))
        print("Trained model loaded:", predictive_model_path)
        predictive_model.to(device)
        predictive_model.eval()

        policy_model = TD3.load(policy_model_path)

        print("Loading predictor's training data!")
        memory_rb = torch.load(optimal_memory_path, map_location=device)
        states_min, states_max = states_min_max_finder(memory_rb)
        # prevent division by zero in normalization
        no_need_normalization = np.where((states_min == states_max))[0]

        individual_feature_auc = {}
        for h in args.horizons:
            individual_feature_auc[h] = {}
            for f in range(num_features):
                individual_feature_auc[h][f] = []
        for h in args.horizons:
            for ss in args.sampling_sizes:
                all_avg_aucs = []
                all_max_aucs = []
                all_avg_false_alarm_rates = []
                all_max_false_alarm_rates = []
                on_features_original_cusums = []
                on_scores_original_cusums = []
                for _ in range(args.n_eval_episodes):
                    random_seed = random.randint(0, 1000)
                    env = DummyVecEnv(
                        [env_preparation.make_env(args.env_name, 0, random_seed, wrapper_class=env_preparation.TimeFeatureWrapper,
                                                  env_kwargs={'power': args.anomalous_power,
                                                              'anomaly_injection': args.anomaly_injection,
                                                              'case': args.case})])
                    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=args.clip_obs)
                    # do not update them at test time
                    env.training = False
                    # reward normalization is not needed at test time
                    env.norm_reward = False

                    mean_reward, std_reward, observations = evaluate_policy(policy_model, env, n_eval_episodes=1)
                    # print(f"Best model mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

                    # normalizing data
                    tmp_observations = ((np.array(observations[0])[:, :, :num_features] - states_min) / (states_max - states_min))
                    normalized_observations = [np.concatenate((tmp_observations, np.array(observations[0])[:, :, num_features].reshape(-1, 1, 1)), axis=2)]
                    normalized_observations = np.array(normalized_observations)
                    for index in no_need_normalization:
                        normalized_observations[0, :, 0, index] = states_min[index]

                    dists_per_horizon = {}

                    when_anomaly_occurred = np.zeros(len(normalized_observations[0]) - h)
                    when_anomaly_occurred[args.anomaly_injection - h:] = 1
                    sep_features_r, true_r, dist_r, merg_avg_auc, merg_max_auc, ass,\
                        avg_f_a_rate, max_f_a_rate = ar_anomaly_detection(predictive_model, args.gru_units, args.num_tau_sample,
                                                                       args.sampling_sizes[0], device, num_features, normalized_observations[0],
                                                                       h, when_anomaly_occurred, args.is_recurrent_v2)
                    all_avg_aucs.append(merg_avg_auc)
                    all_max_aucs.append(merg_max_auc)
                    all_avg_false_alarm_rates.append(avg_f_a_rate)
                    all_max_false_alarm_rates.append(max_f_a_rate)
                    on_features_cusum_changepoints = original_cusum(true_r, num_features)
                    on_scores_cusum_changepoints = original_cusum(np.array(ass), num_features)
                    on_features_original_cusums.append(on_features_cusum_changepoints[0])
                    on_scores_original_cusums.append(on_scores_cusum_changepoints[0])
                    dists_per_horizon[h] = dist_r
                    for f, value in sep_features_r.items():
                        individual_feature_auc[h][f].append(value[3])
                warmup = 0
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