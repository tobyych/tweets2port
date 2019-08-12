import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from brokenaxes import brokenaxes
import markowitz as m
import pandas as pd
import data as d


def plot_frontier(model_dict, path_to_output="./output/graphs/"):
    fig, ax = plt.subplots(1)
    # bax = brokenaxes(ylims=((0, 0.01), (0.05, 0.4)), d=0.001, tilt=0, hspace=0.5)
    for model_name, line_color in model_dict.items():
        path_to_pickle = os.path.join("./temp/pred", model_name, "pred.pickle")
        if os.path.exists(path_to_pickle):
            returns = pd.read_pickle(path_to_pickle)
        else:
            returns = d.load_predictions(
                path_to_pred=os.path.join("./output", model_name),
                path_to_output=os.path.join("./temp/pred", model_name),
            )
        if model_name == "actual":
            print(returns)
            _, act_mean, act_std = m.equally_weighted_port(returns)
            ax.plot(
                act_std,
                act_mean,
                color=line_color,
                marker=".",
                markersize=5,
                label="equal weighted",
            )
            plt.axhline(y=act_mean, color=line_color, ls="--")
            plt.axvline(x=act_std, color=line_color, ls="--")
        else:
            _, temp_mean, temp_std = m.optimal_port(returns, short_sell=False)
            ax.plot(temp_std, temp_mean, color=line_color, label=model_name)
    # plt.ylim(0, 0.4)
    # plt.xlim(0, 0.3)
    plt.ylabel("mean")
    plt.xlabel("std")
    plt.legend()
    plt.show()
    if not os.path.exists("./output/graphs"):
        os.makedirs("./output/graphs")
    fig.savefig(os.path.join(path_to_output, "frontier-no-short-sell-v2.png"))


def plot_etf_points(mean_var_dict):
    fig, ax = plt.subplots()
    for etf_name, mean_var_tuple in mean_var_dict.items():
        ax.plot(
            mean_var_tuple[1],
            mean_var_tuple[0],
            marker=".",
            markersize=6,
            label=etf_name,
        )
    plt.ylim(-0.015, 0.01)
    plt.xlim(0, 0.1)
    plt.ylabel("expected return")
    plt.xlabel("standard deviation")
    plt.legend()
    plt.title("Actual mean and variance of top traded ETFs")
    plt.show()
    fig.savefig("output/graphs/etf.png")
    return fig
