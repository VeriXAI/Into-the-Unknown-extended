import shutil

import pandas as pd
import json
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import os


style_dic = {'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.grid': False,
                      'axes.axisbelow': 'line', 'axes.labelcolor': 'black', 'figure.facecolor': (1, 1, 1, 0),
                      'grid.color': '#b0b0b0', 'grid.linestyle': '-', 'text.color': 'black',
                      'xtick.color': 'black', 'ytick.color': 'black', 'xtick.direction': 'in',
                      'ytick.direction': 'in',
                      'patch.edgecolor': 'black', 'patch.force_edgecolor': False,
                      'image.cmap': 'viridis', 'font.family': ['sans-serif'],
                      'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif'],
                      'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False,
                      'axes.spines.left': False, 'axes.spines.bottom': True, 'axes.spines.right': True,
                      'axes.spines.top': True}
contex_dic = {'font.size': 16.0, 'axes.labelsize': 'medium', 'axes.titlesize': 'large',
               'xtick.labelsize': 'medium', 'ytick.labelsize': 'medium', 'legend.fontsize': 'medium',
               'axes.linewidth': 0.8, 'grid.linewidth': 0.8, 'lines.linewidth': 1.5, 'lines.markersize': 6.0,
               'patch.linewidth': 1.0, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
               'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6, 'xtick.major.size': 3.5,
               'ytick.major.size': 3.5, 'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
               'legend.title_fontsize': None}

def compute_percentages(df):
    s = (df['tp'] + df['fn'] + df['fp'] + df['tn'])
    df2 = pd.DataFrame()
    df2['tpp'] = df['tp'] / s
    df2['fpp'] = df['fp'] / s
    df2['tnp'] = df['tn'] / s
    df2['fnp'] = df['fn'] / s
    return df2.mean()


def compute_fpr_tpr(df):
    df2 = pd.DataFrame()
    df2['tpr'] = df['tp'] / (df['tp'] + df['fn'])
    df2['fpr'] = df['fp'] / (df['fp'] + df['tn'])
    return df2.mean()


def compute_accuracy(df):
    df2 = pd.DataFrame()
    df2['accuracy'] = (df['tp'] + df["tn"]) / (df['tp'] + df['fn'] + df['fp'] + df['tn'])
    return df2


def compute_accuracy_mean(df):
    df2 = pd.DataFrame()
    df2['accuracy'] = (df['tp'] + df["tn"]) / (df['tp'] + df['fn'] + df['fp'] + df['tn'])
    return df2.mean()


def rate_transform(path):
    print("Open:", path)
    with open(path, 'r') as f:
        dic = json.load(f)
    rate_dic = {x: {k: [] for k in dic[x][0][1]["monitor"].keys()} for x in ["abstraction", "base"]}
    for i in range(len(dic["abstraction"])):
        for x in ["abstraction", "base"]:
            for k, v in dic[x][i][1]["monitor"].items():
                rate_dic[x][k].append(v)
    return {(k1, k2): v for k1 in rate_dic.keys() for k2, v in rate_dic[k1].items()}


def get_dict(name):
    with open(name, 'r') as f:
        dic = json.load(f)
    return dic


def save_dict(name, val):
    with open(name, 'w') as f:
        json.dump(val, f)


def evaluate(name, data_path, sum_path):
    dic = rate_transform(data_path)
    df = pd.DataFrame().from_dict(dic)
    s = []
    s.append("------------------------------------------------")
    s.append('Abstraction')
    s.append("------------------------------------------------")
    s.append(compute_percentages(df['abstraction']))
    s.append("------------------------------------------------")
    s.append(compute_fpr_tpr(df['abstraction']))
    s.append(compute_accuracy_mean(df['abstraction']))
    s.append("------------------------------------------------")
    s.append("")
    s.append("")
    s.append("------------------------------------------------")
    s.append('Base')
    s.append("------------------------------------------------")
    s.append(compute_percentages(df['base']))
    s.append("------------------------------------------------")
    s.append(compute_fpr_tpr(df['base']))
    s.append(compute_accuracy_mean(df['base']))
    s.append("------------------------------------------------")
    s.append("")
    s.append("")
    s.append("------------------------------------------------")
    s.append('Relative Values (Abstraction / Base)')
    s.append("------------------------------------------------")
    s.append(compute_percentages(df['abstraction']) / compute_percentages(df['base']) - 1)
    s.append("------------------------------------------------")
    s.append(compute_fpr_tpr(df['abstraction']) / compute_fpr_tpr(df['base']) - 1)
    s.append(compute_accuracy_mean(df['abstraction']) / compute_accuracy_mean(df['base']) - 1)
    s = '\n'.join([str(i) for i in s])
    with open(os.path.join(sum_path, name.split(".")[0] + "_summary.txt"), "w") as f:
        f.write(s)


def rel_tpr_fpr(path):
    dic = rate_transform(path)
    df = pd.DataFrame().from_dict(dic)
    return compute_fpr_tpr(df['abstraction']) / compute_fpr_tpr(df['base']) - 1


def rel_accuracy(path):
    dic = rate_transform(path)
    df = pd.DataFrame().from_dict(dic)
    return compute_accuracy(df['abstraction']) / compute_accuracy(df['base']) - 1


def rate_transform_nn(path):
    with open(path, 'r') as f:
        dic = json.load(f)
    rate_dic = {x: {k: [] for k in dic[x][0][1]["network"].keys()} for x in ["abstraction", "base"]}
    for i in range(len(dic["abstraction"])):
        for x in ["abstraction", "base"]:
            for k, v in dic[x][i][1]["network"].items():
                rate_dic[x][k].append(v[1])
    return {(k1, k2): v for k1 in rate_dic.keys() for k2, v in rate_dic[k1].items()}


def rel_accuracy_nn(path):
    dic = rate_transform_nn(path)
    df = pd.DataFrame().from_dict(dic)
    return df['abstraction'] / df['base'] - 1


def plot_accuracy_alpha(name, data_dir, fig_dir):
    epoch = -1
    data = []
    print(data_dir)
    for i, l in enumerate(get_dict(data_dir)["training_log"]):
        for j in l[1]:
            if epoch != j["epoch"]:
                epoch = j["epoch"]
                data.append([i, j["epoch"] + 4, "alpha", j["alpha"]])
                data.append([i, j["epoch"] + 4, "accuracy", j["accuracy"]])
    df = pd.DataFrame(data, columns=["Experiment", "Epoch", "", "Value"])
    sns.relplot(x="Epoch", y="Value", hue="", kind="line", style="", markers=True, data=df, row="Experiment")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, name.split(".")[0] + "_alpha.png"), dpi=500, bbox_inches="tight", transparent=True)


def list_experiments(path):
    dic = get_dict(path)
    for i, e in enumerate(dic["abstraction"]):
        print(i + 1, e[0])


def collect_data_fpr_tpr(path, name):
    data = []
    for p in sorted(os.listdir(path), key=lambda x: (x[0], x[1])):
        if p[0] != ".":
            experiment = p.split("_")[-1].split(".")[0]
            res = rel_tpr_fpr(os.path.join(path,p))
            data.append([name, experiment, "TPR", res["tpr"] * 100])
            data.append([name, experiment, "FPR", res["fpr"] * 100])

    return data


def collect_data_accuracy(path, name):
    data = []
    for p in sorted(os.listdir(path), key=lambda x: (x[0], x[1])):
        if p[0] != ".":
            experiment = p.split("_")[-1].split(".")[0]
            res = rel_accuracy(os.path.join(path,p))
            for c in res:
                for r in res[c]:
                    data.append([name, experiment, "Accuracy", r * 100])
    return data


def collect_data_accuracy_mean(path, name):
    data = []
    for p in sorted(os.listdir(path), key=lambda x: (x[0], x[1])):
        if p[0] != ".":
            experiment = p.split("_")[-1].split(".")[0]
            res = rel_accuracy(os.path.join(path,p)).mean()
            data.append([name, experiment, "Accuracy", res["accuracy"] * 100])
    return data


def collect_data_accuracy_mean_nn(path, name):
    data = []
    for p in sorted(os.listdir(path), key=lambda x: (x[0], x[1])):
        if p[0] != ".":
            experiment = p.split("_")[-1].split(".")[0]
            res = rel_accuracy(os.path.join(path,p)).mean()
            res2 = rel_accuracy_nn(os.path.join(path,p)).mean()
            data.append([name, experiment, "Monitor", res["accuracy"] * 100])
            data.append([name, experiment, "Network", res2["accuracy"] * 100])
    return data


def collect_overview_data(directory):
    metrics = [("fpr-tpr", collect_data_fpr_tpr), ("accuracy", collect_data_accuracy_mean_nn)]
    data_dic = {}
    for k in metrics:
        n, foo = k
        data = []
        for p in sorted(os.listdir(directory)):
            path = os.path.join(directory, p)
            if os.path.isdir(path) and p[0] != ".":
                data += foo(path, p)
        data_dic[n] = pd.DataFrame(data, columns=["Dataset", "Experiment", "Metric", "Percent"])
    return data_dic


def plot_fpr_tpr_bar(path, df):
    sns.set(style=style_dic, context=contex_dic, palette="tab10")
    ax = sns.catplot(x="Experiment", y="Percent", hue="Metric", col="Dataset", kind="bar", data=df["fpr-tpr"],
                     order=["1a", "1b", "1c", "2a", "2c", "3c"], legend_out=False,
                     col_order=["MNIST", "FMNIST", "GTSRB"],
                     hue_order=["TPR", "FPR"], legend=False
                     )
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "fpr_tpr_bar.png"), dpi=500, bbox_inches="tight", transparent=True)


def plot_accuracy_bar(path, df):
    df = df.copy()
    df["accuracy"] = df["accuracy"].rename(columns={"Metric": "Accuracy"})
    sns.set(style=style_dic, context=contex_dic, palette="tab10")
    ax = sns.catplot(x="Experiment", y="Percent", hue="Accuracy", col="Dataset", kind="bar", data=df["accuracy"],
                     order=["1a", "1b", "1c", "2a", "2c", "3c"], legend_out=False,
                     col_order=["MNIST", "FMNIST", "GTSRB"],
                     hue_order=["Monitor", "Network"], legend=False
                     )
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "accuracy_bar.png"), dpi=500, bbox_inches="tight", transparent=True)


def plot_training_data(data_dir, fig_dir):
    for p in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, p)
        if os.path.isdir(path) and p[0] != ".":
            for q in sorted(os.listdir(path)):
                data_path = os.path.join(path, q)
                if ".json" in q:
                    plot_accuracy_alpha(p + "_" + q, data_path, fig_dir)


def plot_figures(data_dir, fig_dir):
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.mkdir(fig_dir)
    df = collect_overview_data(data_dir)
    plot_fpr_tpr_bar(fig_dir, df)
    plot_accuracy_bar(fig_dir, df)
    plot_training_data(data_dir, fig_dir)


def generate_summary(data_dir, sum_dir):
    if os.path.exists(sum_dir):
        shutil.rmtree(sum_dir)
    os.mkdir(sum_dir)
    for p in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, p)
        if os.path.isdir(path) and p[0] != ".":
            for q in sorted(os.listdir(path)):
                data_path = os.path.join(path, q)
                if ".json" in q:
                    evaluate(p + "_" + q, data_path, sum_dir)


def analyse(directory):
    data_dir = os.path.join(directory, "data")
    fig_dir = os.path.join(directory, "figure")
    sum_dir = os.path.join(directory, "summary")
    plot_figures(data_dir, fig_dir)
    generate_summary(data_dir, sum_dir)
