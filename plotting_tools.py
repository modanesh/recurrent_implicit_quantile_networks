import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

sns.set()

color_codes = {'1': 'brown', '2': 'darkred', '3': 'red', '4': 'salmon', '5': 'orangered', '6': 'sienna',
               '7': 'saddlebrown', '8': 'sandybrown', '9': 'peru', '10': 'darkorange', '11': 'orange',
               '12': 'goldenrod', '13': 'gold', '14': 'khaki', '15': 'darkkhaki', '16': 'olive', '17': 'yellow',
               '18': 'yellowgreen', '19': 'chartreuse', '20': 'lightgreen', '21': 'darkgreen', '22': 'lime',
               '23': 'springgreen', '24': 'turquoise', '25': 'darkslategrey', '26': 'cyan', '27': 'lightblue',
               '28': 'deepskyblue', '29': 'steelblue', '30': 'dodgerblue', '31': 'slategrey', '32': 'royalblue',
               '33': 'navy', '34': 'blue', '35': 'indigo', '36': 'darkviolet', '37': 'plum', '38': 'magenta',
               '39': 'hotpink', '40': 'pink'}


def plot_methods_roc(all_results, dir, env_name, method, gvd_names, gamma, target_type, bootstrap, is_recurrent=False, horizon=0):
    num_distinct_gvds = len(gvd_names)
    fig, axs = plt.subplots(math.ceil(num_distinct_gvds / 2), 2, figsize=(10, 14))
    r, c = 0, 0
    for key in all_results.keys():
        if key.__contains__("cartlocation") or key.__contains__("cos1") or key.__contains__("posx"):
            axs[0, 0].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("cartvelocity") or key.__contains__("sin1") or key.__contains__("posy"):
            axs[0, 1].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("polelocation") or key.__contains__("cos2") or key.__contains__("velocityx"):
            axs[1, 0].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("polevelocity") or key.__contains__("sin2") or key.__contains__("velocityy"):
            axs[1, 1].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("velocity1") or key.__contains__("angle"):
            axs[2, 0].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("velocity2") or key.__contains__("angvelocity"):
            axs[2, 1].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("leftleg"):
            axs[3, 0].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
        elif key.__contains__("rightleg"):
            axs[3, 1].plot(all_results[key][0], all_results[key][1],
                           label="H:" + key.split("_")[-1] + " - AUC:" + str(round(all_results[key][3], 2)),
                           color=color_codes[key.split("_")[-1]])
    for i in range(num_distinct_gvds):
        axs[r, c].plot(np.arange(2), np.arange(2), label="Random", color='purple')
        axs[r, c].set(xlabel='FPR', ylabel='TPR')
        axs[r, c].set_title("GVD: " + gvd_names[i].split("_")[0])
        axs[r, c].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        if r < math.ceil(num_distinct_gvds / 2) - 1:
            r += 1
        else:
            c += 1
            r = 0
    if not is_recurrent:
        fig.suptitle("ROC Curve\ngamma: " + str(gamma) + "\npr type: " + target_type)
        fig.tight_layout()
        fig.savefig(os.path.join(dir, "ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_gamma_" + str(gamma) + ".png"))
    else:
        if bootstrap:
            fig.suptitle("ROC Curve\nhorizon: " + str(horizon) + "\npr type: " + target_type + " - bootstrapping")
            fig.tight_layout()
            fig.savefig(os.path.join(dir, "ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizon) + "_bootstrap.png"))
        else:
            fig.suptitle("ROC Curve\nhorizon: " + str(horizon) + "\npr type: " + target_type)
            fig.tight_layout()
            fig.savefig(os.path.join(dir, "ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizon) + ".png"))
    # fig.show()
    plt.clf()
    plt.cla()
    plt.close()
    print("ROC AUC plot saved in", dir)


def plot_combined_roc(results, dir, env_name, method, gamma, target_type, bootstrap, merging_method, is_recurrent=False, horizon=0):
    fig, axs = plt.subplots(figsize=(4, 5))
    for key in results.keys():
        axs.plot(results[key][0], results[key][1],
                 label="H:" + key.split("_")[-1] + " - AUC:" + str(round(results[key][3], 2)),
                 color=color_codes[key.split("_")[-1]])
    for i in range(len(results)):
        if i == 0:
            axs.plot(np.arange(2), np.arange(2), label="Random", color='purple')
        axs.set(xlabel='FPR', ylabel='TPR')
        axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

    if not is_recurrent:
        fig.suptitle("Combined ROC Curve\ngamma: " + str(gamma) + "\npr type: " + target_type)
        fig.tight_layout()
        fig.savefig(os.path.join(dir, "Combined_ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_gamma_" + str(gamma) + ".png"))
    else:
        if bootstrap:
            fig.suptitle("Combined ROC Curve\nhorizon: " + str(horizon) + "\npr type: " + target_type + " - bootstrapping" + "\ncombining method: " + merging_method)
            fig.tight_layout()
            fig.savefig(os.path.join(dir, "Combined_ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizon) + "_" + merging_method + "_bootstrap.png"))
        else:
            fig.suptitle("Combined ROC Curve\nhorizon: " + str(horizon) + "\npr type: " + target_type + "\ncombining method: " + merging_method)
            fig.tight_layout()
            fig.savefig(os.path.join(dir, "Combined_ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizon) + "_" + merging_method + ".png"))
    # fig.show()
    plt.clf()
    plt.cla()
    plt.close()
    print("Combined ROC AUC plot saved in", dir)


def aucs_progress(all_results, dir, env_name, method, gamma):
    for key, value in all_results.items():
        plt.plot(value[-1], label=key)

    plt.xlabel("Horizon")
    plt.ylabel("AUC")
    plt.title("AUCs progress\ngamma: " + str(gamma))
    plt.legend()
    plt.savefig(os.path.join(dir, "AUC_progress_" + method + "_" + env_name + "_gamma_" + str(gamma) + ".png"))
    plt.clf()
    plt.cla()
    plt.close()


def update_plot(i, ax, value_dist, min_val, max_val):
    label = 'step {0}'.format(i)
    ax.clear()
    ax.set_title(label)
    ax.set_xlabel('Returns')
    ax.set_ylabel('Actions')
    plt.axis([min_val, max_val, -0.5, 1.5])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.scatter(value_dist[i][0], ["left"] * 32)
    # ax.scatter(value_dist[i][1], ["right"] * 32)


def plot_dist(value_dist, save_path):
    for ep in range(len(value_dist)):
        min_val = math.floor(value_dist[ep].min())
        max_val = math.ceil(value_dist[ep].max())
        fig, ax = plt.subplots(figsize=(8, 2))
        fig.set_tight_layout(True)
        anim = FuncAnimation(fig, update_plot, frames=np.arange(0, value_dist[ep].shape[0]), fargs=(ax, value_dist[ep], min_val, max_val), interval=100)
        anim.save(save_path + '/q_dist_ep' + str(ep) + '.gif',  writer='imagemagick')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(np.mean(value_dist[ep][:,0,:], axis=1), label="Left")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Mean return")
        # ax.plot(np.mean(value_dist[ep][:,1,:], axis=1), label="Right")
        ax.legend()
        plt.savefig(save_path + "/mean_returns_ep" + str(ep) + ".png")
        plt.close(fig)


def plot_histogram_array(data, label):
    x = data.numpy()
    y = np.zeros(data.shape[0])

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.25
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)

    # the scatter plot:
    ax_scatter.scatter(x, y, label=label, color='b')

    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-1, 1))
    ax_scatter.set_xlabel("values")
    ax_scatter.legend()

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='b')
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histx.set_ylabel("count")

    plt.show()


def plot_scores(scores, anomaly_appeared, dir, env_name, method, gamma, horizons, bootstrapped, is_recurrent, target_type):
    for key, value in anomaly_appeared.items():
        plt.axvline(x=value.index(1), label="Anomaly started (H={})".format(key))

    counter = 1
    summed_value = np.zeros(len(scores[next(iter(scores))][0]))
    for key, value in scores.items():
        summed_value += np.array(value[0])
        plt.plot(value[0], color_codes[str(counter)], label=key.split("_")[0] + " - H=" + key.split("_")[-1])
        counter += 7
    plt.plot(summed_value, color_codes[str(counter)], label="Summation")
    # counter = 0
    # for key, value in scores.items():
    #     smoothed_value = array_smoothie(value[0], 0.9)
    #     plt.plot(smoothed_value, color_codes[str(counter + 1)], label=key.split("_")[0])
    #     counter += 2

    plt.xlabel("Step")
    plt.ylabel("Anomaly score")
    if not is_recurrent:
        plt.title("Anomaly score\ngamma: " + str(gamma))
        plt.legend()
        plt.savefig(os.path.join(dir, "Anomaly_scores_" + method + "_" + env_name + "_gamma_" + str(gamma) + ".png"))
    else:
        if not bootstrapped:
            plt.title("Anomaly score\nhorizon: " + str(horizons) + "\npr type: " + target_type)
            plt.legend()
            plt.savefig(os.path.join(dir, "Anomaly_scores_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizons) + ".png"))
        else:
            plt.title("Anomaly score\nhorizon: " + str(horizons) + "\npr type: " + target_type + " - bootstrapping")
            plt.legend()
            plt.savefig(os.path.join(dir, "Anomaly_scores_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizons) + "_bootstrap.png"))
    plt.clf()
    plt.cla()
    plt.close()


def array_smoothie(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed


def plot_heatmaps(all_aucs, gvd_names, horizons, gammas, dir, env_name, ad_algo, anomaly_type):
    x_axis_labels = horizons
    y_axis_labels = gammas

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    cartlocations, cartvelocities, polelocations, polevelocities = all_aucs
    # cartlocations: (cls0, cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9)
    sns.heatmap(cartlocations, cbar=False, ax=axs[0,0], xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidth=0.15, cmap="YlGnBu", vmin=0, vmax=1)
    sns.heatmap(cartvelocities, cbar=False, ax=axs[0,1], xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidth=0.15, cmap="YlGnBu", vmin=0, vmax=1)
    sns.heatmap(polelocations, cbar=False, ax=axs[1,0], xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidth=0.15, cmap="YlGnBu", vmin=0, vmax=1)
    sns.heatmap(polevelocities, cbar=False, ax=axs[1,1], xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidth=0.15, cmap="YlGnBu", vmin=0, vmax=1)

    r, c = 0, 0
    for i in range(len(all_aucs)):
        axs[r, c].set_xlabel("Horizon")
        axs[r, c].set_ylabel("Gamma")
        axs[r, c].invert_yaxis()
        axs[r, c].set_title("GVD: " + gvd_names[i])
        if r < 2 - 1:
            r += 1
        else:
            c += 1
            r = 0

    im = plt.gca().get_children()[0]
    cax = fig.add_axes([0.93, 0.3, 0.02, 0.5])
    fig.colorbar(im, cax=cax)
    fig.subplots_adjust(hspace=.3)

    plt.suptitle("Horizon and gamma effects on AUC (AD: KNN)")
    plt.show()
    plt.savefig(os.path.join(dir, "heatmaps_" + env_name + "_H_lambda_" + ad_algo + "_" + anomaly_type + "anomaly" + ".png"))


def plot_losses(train_loss, test_loss, result_folder, horizon, gvd_name, info, bootstrapped):
    plt.plot(train_loss, label="training loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    if not bootstrapped:
        plt.savefig(os.path.join(result_folder, "losses_" + gvd_name + "_" + info + "_h" + str(horizon) + ".png"))
    else:
        plt.savefig(os.path.join(result_folder, "losses_" + gvd_name + "_" + info + "_h" + str(horizon) + "_bootstrap.png"))
    plt.clf()


def rgvd_accuracy(results, result_folder, horizon, info, bootstrapped):
    fig, axs = plt.subplots(int(len(results) / 2), 2, figsize=(12, 10))
    r, c = 0, 0
    for key in results.keys():
        axs[r, c].plot(results[key][1], color='limegreen')
        axs[r, c].plot(results[key][0], color='teal')
        axs[r, c].set(xlabel='step', ylabel='return')
        axs[r, c].set_title("GVD: " + key.split("_")[0])
        labels = ["actual MC returns", "rGVD returns"]
        axs[r, c].legend(labels=labels, loc='upper right', labelcolor=['teal', 'limegreen'], handlelength=0)
        if r < math.ceil(len(results) / 2) - 1:
            r += 1
        else:
            c += 1
            r = 0
    # fig.show()
    if not bootstrapped:
        fig.suptitle("Recurrent GVD accuracy\nhorizon: " + str(horizon))
        fig.tight_layout()
        fig.savefig(os.path.join(result_folder, "rGVD_accuracy_" + info + "_h" + str(horizon) + ".png"))
    else:
        fig.suptitle("Recurrent GVD accuracy\nhorizon: " + str(horizon) + "\nbootstrapped")
        fig.tight_layout()
        fig.savefig(os.path.join(result_folder, "rGVD_accuracy_" + info + "_h" + str(horizon) + "_bootstrap.png"))


def plot_online_anomalies(scores, real_anomaly, R, horizons, result_folder, method, env_name, target_type, bootstrap, Nw=10):
    fig, axs = plt.subplots(len(horizons) + 1, figsize=(12, 10))
    for h in horizons:
        axs[0].plot(scores, color='red', label='anomaly scores')
        axs[0].set_ylabel("Anomaly scores")
        actual_ep_len = len(R[Nw, Nw+1:])
        axs[1].plot(real_anomaly[str(h)][:actual_ep_len], color='limegreen', label='real anomalies in env')
        axs[1].plot(R[Nw, Nw+1:], label='detected changepoints (bocd)')
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Changepoint probability")

    if bootstrap:
        fig.suptitle("Online Anomaly Detection\nhorizon: " + str(horizons) + "\npr type: " + target_type + " - bootstrapping")
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(result_folder, "Online_AD_" + method + "_" + env_name + "_" + target_type + "_h_" +
                                 str(horizons) + "_bootstrap.png"))
    else:
        fig.suptitle("Online Anomaly Detection\nhorizon: " + str(horizons) + "\npr type: " + target_type)
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(result_folder, "Online_AD_" + method + "_" + env_name + "_" + target_type + "_h_" +
                                 str(horizons) + ".png"))

