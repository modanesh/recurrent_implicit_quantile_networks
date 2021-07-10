import gym
import time
import math
import torch
import numpy as np
import sklearn.metrics
from gym import wrappers
import autoregressive_control
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor


def feature_index_extractor(gvd_name):
    if gvd_name.__contains__("cartlocation") or gvd_name.__contains__("cos1") or gvd_name.__contains__("posx"):
        feature_index = 0
    elif gvd_name.__contains__("cartvelocity") or gvd_name.__contains__("sin1") or gvd_name.__contains__("posy"):
        feature_index = 1
    elif gvd_name.__contains__("polelocation") or gvd_name.__contains__("cos2") or gvd_name.__contains__("velocityx"):
        feature_index = 2
    elif gvd_name.__contains__("polevelocity") or gvd_name.__contains__("sin2") or gvd_name.__contains__("velocityy"):
        feature_index = 3
    elif gvd_name.__contains__("velocity1") or gvd_name.__contains__("angle"):
        feature_index = 4
    elif gvd_name.__contains__("velocity2") or gvd_name.__contains__("angvelocity"):
        feature_index = 5
    elif gvd_name.__contains__("leftleg"):
        feature_index = 6
    elif gvd_name.__contains__("rightleg"):
        feature_index = 7
    return feature_index


def expected_likelihood(distribution, expectation):
    score = round(abs(distribution.mean() - expectation).item(), 5)
    return score


def histogram_likelihood(distribution, expectation):
    mu = round(distribution.mean(), 5)
    sigma = round(distribution.std(), 5)

    ranges = {}
    for i in range(len(distribution)):
        if sigma != 0:
            if mu - sigma <= distribution[i] < mu + sigma and (mu - sigma, mu + sigma) not in ranges:
                ranges[(mu - sigma, mu + sigma)] = 0
            elif mu + sigma <= distribution[i] < mu + 2 * sigma and (mu + sigma, mu + 2 * sigma) not in ranges:
                ranges[(mu + sigma, mu + 2 * sigma)] = 0
            elif mu - 2 * sigma <= distribution[i] < mu - sigma and (mu - 2 * sigma, mu - sigma) not in ranges:
                ranges[(mu - 2 * sigma, mu - sigma)] = 0
            elif mu + 2 * sigma <= distribution[i] < mu + 3 * sigma and (mu + 2 * sigma, mu + 3 * sigma) not in ranges:
                ranges[(mu + 2 * sigma, mu + 3 * sigma)] = 0
            elif mu - 3 * sigma <= distribution[i] < mu - 2 * sigma and (mu - 3 * sigma, mu - 2 * sigma) not in ranges:
                ranges[(mu - 3 * sigma, mu - 2 * sigma)] = 0

            if mu - sigma <= distribution[i] < mu + sigma:
                ranges[(mu - sigma, mu + sigma)] += 1
            elif mu + sigma <= distribution[i] < mu + 2 * sigma:
                ranges[(mu + sigma, mu + 2 * sigma)] += 1
            elif mu - 2 * sigma <= distribution[i] < mu - sigma:
                ranges[(mu - 2 * sigma, mu - sigma)] += 1
            elif mu + 2 * sigma <= distribution[i] < mu + 3 * sigma:
                ranges[(mu + 2 * sigma, mu + 3 * sigma)] += 1
            elif mu - 3 * sigma <= distribution[i] < mu - 2 * sigma:
                ranges[(mu - 3 * sigma, mu - 2 * sigma)] += 1

    found_likelihood = False
    for key, value in ranges.items():
        if key[0] <= expectation < key[1]:
            score = value / len(distribution)
            found_likelihood = True
    if not found_likelihood:
        score = 0
        if mu == expectation:
            score = 1

    return score


def local_outlier_factor(distribution, actual_return):
    lof = LocalOutlierFactor(n_neighbors=8)
    lof.fit_predict(np.append(distribution, actual_return).reshape(-1, 1))
    score = abs(lof.negative_outlier_factor_[-1])
    return score


def k_nearest_neighbors(distribution, actual_return):
    neigh = NearestNeighbors(n_neighbors=8)
    neigh.fit(distribution.reshape(-1, 1))
    distances, indices = neigh.kneighbors(np.array(actual_return).reshape(-1, 1))
    return distances.sum()


def isolation_forest(distribution, actual_return):
    clf = IsolationForest(n_estimators=10, contamination=0.03)
    clf.fit(distribution.reshape(-1, 1))
    score = abs(clf.score_samples(np.array(actual_return).reshape(-1, 1)))[0]
    return score
    # return 0


def oneclass_svm(distribution, actual_return):
    clf = OneClassSVM(gamma='scale', nu=0.03)
    clf.fit(distribution.reshape(-1, 1))
    score = clf.score_samples(np.array(actual_return).reshape(-1, 1))[0]
    return score


def anomaly_score_undiscounted(transitions, starting_action, gvd_model, batch_size, num_quantile_sample, as_method, h_s, gvd_name, type):
    starting_state = torch.Tensor(transitions[0]).unsqueeze(0)
    tau = torch.Tensor(np.random.rand(batch_size * num_quantile_sample, 1))
    value_dist, h_s = gvd_model(starting_state, h_s, tau, num_quantile_sample)
    value_dist = value_dist.squeeze(0)[starting_action]

    feature_index = feature_index_extractor(gvd_name)

    if type == "delta":
        expected_value = np.sum(np.diff(np.array(transitions), axis=0), axis=0)[feature_index]
    elif type == "abs_delta":
        expected_value = np.sum(abs(np.diff(np.array(transitions), axis=0)), axis=0)[feature_index]
    elif type == "time_avg":
        expected_value = np.sum(np.diff(np.array(transitions), axis=0), axis=0)[feature_index] / (len(transitions) - 1)
    else:
        assert False, "Undefined/unknown method given to calculate the target return for GVDs. Notice the given arguments!"

    if as_method == "histogram":
        norm_score = histogram_likelihood(value_dist.cpu().numpy(), round(expected_value, 5))
    elif as_method == "lof":
        norm_score = local_outlier_factor(value_dist.cpu().numpy(), round(expected_value, 5))
    elif as_method == "knn":
        norm_score = k_nearest_neighbors(value_dist.cpu().numpy(), round(expected_value, 5))
    elif as_method == "iforest":
        norm_score = isolation_forest(value_dist.cpu().numpy(), round(expected_value, 5))
    elif as_method == "svm":
        norm_score = oneclass_svm(value_dist.cpu().numpy(), round(expected_value, 5))
    else:
        assert False, "Anomaly score measuring method is not given properly! Check '--score_calc_method'!"
    return norm_score, h_s


def process_anomalies(args, env, all_gvd_models, main_model, epsilon, gamma, horizons, num_iteration, type):
    batch_size = 1
    total_reward = []
    all_scores = {}
    all_scores_merged = {}

    for h in horizons:
        all_scores_merged[str(h)] = []
        for _, gvd_name in all_gvd_models:
            all_scores[gvd_name + "_" + str(h)] = []

    with torch.no_grad():
        for ep in range(num_iteration):
            state = env.reset()
            state = torch.Tensor(state).unsqueeze(0)
            done, ep_reward = False, 0
            all_transitions, all_actions, ep_scores = {}, {}, {}
            h_s_dict = {}
            for _, gvd_name in all_gvd_models:
                all_transitions[gvd_name] = []
                all_actions[gvd_name] = []
                h_s_dict[gvd_name.split("_0")[0]] = torch.zeros(args.num_quantile_sample, args.gru_units)
                for h in horizons:
                    ep_scores[gvd_name + "_" + str(h)] = []
            while not done:
                action, z_values = autoregressive_control.get_action(state, main_model, epsilon, env, args.num_quantile_sample)

                next_state, reward, done, _ = env.step(action)
                next_state = torch.Tensor(next_state).unsqueeze(0)
                ep_reward += reward

                for gvd_model, gvd_name in all_gvd_models:
                    all_transitions[gvd_name].append(state.numpy().reshape(-1))
                    all_actions[gvd_name].append(action)
                    for h in horizons:
                        if len(all_transitions[gvd_name]) > h:
                            gvd_model = [all_gvd_models[x][0] for x in range(len(all_gvd_models)) if all_gvd_models[x][1] == gvd_name][0]
                            if args.recurrent_gvd:
                                n_score, h_s = anomaly_score_undiscounted(all_transitions[gvd_name][-h - 1:],
                                                                          all_actions[gvd_name][-h - 1:][0], gvd_model,
                                                                          batch_size, args.num_quantile_sample,
                                                                          args.score_calc_method,
                                                                          h_s_dict[gvd_name.split("_0")[0]], gvd_name, type)
                                h_s_dict[gvd_name.split("_0")[0]] = h_s
                            ep_scores[gvd_name + "_" + str(h)].append(n_score)
                state = next_state
            print("Ep:", str(ep), "reward:", str(ep_reward))
            total_reward.append(ep_reward)
            for key, value in ep_scores.items():
                all_scores[key].append(value)

    if args.merging_method == "avg":
        for key, values in all_scores.items():
            np_values = np.array([np.array(xi) for xi in values])
            if len(all_scores_merged[key.split("_")[-1]]) == 0:
                all_scores_merged[key.split("_")[-1]] = np_values.copy()
            else:
                all_scores_merged[key.split("_")[-1]] += np_values
    elif args.merging_method == "max":
        for h in horizons:
            tmp_placeholder = []
            for key, values in all_scores.items():
                if str(h) == key.split("_")[-1]:
                    tmp_placeholder.extend(values)
            all_scores_merged[str(h)] = np.expand_dims(np.array(tmp_placeholder).max(axis=0), axis=0)

    return all_scores, all_transitions, all_scores_merged


def combined_confusion_matrix(combined_scores, randomness_starts, as_c_method):
    results = {}
    for key, sc in combined_scores.items():
        # Reason behind this part: sklearn.metrics.roc_curve gets (labels, scores) as input arguments. By default it
        # expects to have lower scores for nominal cases and higher scores for anomalous ones. This would be problematic
        # in histogram, iforest, and SVM cases, since in these methods nominal samples have higher scores. Thus, the
        # labels need to be replaced.
        labels = randomness_starts[key.split("_")[-1]]
        if as_c_method == "histogram" or as_c_method == "iforest" or as_c_method == "svm":
            labels = (np.ones(len(labels)) - np.array(labels)).tolist()

        # This part of the code helps with threshold determination in case of histogram method. Using these specified
        # thresholds, a better performance regarding anomaly detection is achieved.
        if as_c_method == "histogram":
            thresholds = []
            tprs, fprs = [], []
            for i in range(1000):
                classifying_threshold = i / 1000
                thresholds.append(classifying_threshold)
                tn_counter, fp_counter = 0, 0
                tp_counter, fn_counter = 0, 0
                for s_i, s in enumerate(sc):
                    if labels[s_i] == 1:
                        if s < classifying_threshold:
                            fp_counter += 1
                        else:
                            tn_counter += 1
                    else:
                        if s < classifying_threshold:
                            tp_counter += 1
                        else:
                            fn_counter += 1
                fpr = round(fp_counter / (tn_counter + fp_counter), 2)
                tpr = round(tp_counter / (tp_counter + fn_counter), 2)
                fprs.append(fpr)
                tprs.append(tpr)
            auc = sklearn.metrics.auc(fprs, tprs)
            results[key] = (fprs, tprs, thresholds, auc)
        else:
            fpr, tpr, thresholds = roc_curve(labels[:len(sc)], sc)
            auc = sklearn.metrics.auc(fpr, tpr)
            results[key] = (fpr, tpr, thresholds, auc)
    return results


def separate_confusion_matrix(nominal_scores, anom_scores, as_c_method):
    results = {}
    for key, nom_value in nominal_scores.items():
        nomi_sc = nom_value[0]
        anom_sc = anom_scores[key]
        scores = np.append(nomi_sc, anom_sc)

        # Reason behind this part: sklearn.metrics.roc_curve gets (labels, scores) as input arguments. By default it
        # expects to have lower scores for nominal cases and higher scores for anomalous ones. This would be problematic
        # in histogram, iforest, and SVM cases, since in these methods nominal samples have higher scores. Thus, the
        # labels need to be replaced.
        if as_c_method == "histogram" or as_c_method == "iforest" or as_c_method == "svm":
            norm_labels = np.ones(len(nomi_sc))
            anorm_labels = np.zeros(len(anom_sc))
        else:
            norm_labels = np.zeros(len(nomi_sc))
            anorm_labels = np.ones(len(anom_sc))

        labels = np.append(norm_labels, anorm_labels)

        # This part of the code helps with threshold determination in case of histogram method. Using these specified
        # thresholds, a better performance regarding anomaly detection is achieved.
        if as_c_method == "histogram":
            nom_value = nom_value[0]
            thresholds = []
            tprs, fprs = [], []
            for i in range(1000):
                classifying_threshold = i / 1000
                thresholds.append(classifying_threshold)
                tn_counter, fp_counter = 0, 0
                tp_counter, fn_counter = 0, 0
                for j in range(len(labels)):
                    if labels[j] == 1:
                        if scores[j] < classifying_threshold:
                            fp_counter += 1
                        else:
                            tn_counter += 1
                    else:
                        if scores[j] < classifying_threshold:
                            tp_counter += 1
                        else:
                            fn_counter += 1
                fpr = round(fp_counter / (tn_counter + fp_counter), 2)
                tpr = round(tp_counter / (tp_counter + fn_counter), 2)
                fprs.append(fpr)
                tprs.append(tpr)
            auc = sklearn.metrics.auc(fprs, tprs)
            results[key] = (fprs, tprs, thresholds, auc)
        else:
            fpr, tpr, thresholds = roc_curve(labels, scores)
            auc = sklearn.metrics.auc(fpr, tpr)
            results[key] = (fpr, tpr, thresholds, auc)
    return results
