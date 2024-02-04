from sb3_contrib import MaskablePPO
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import math
from scipy import stats
from matplotlib.ticker import ScalarFormatter


t = sys.argv[1] if len(sys.argv) >= 2 else "weight"
assert t in ["weight", "bias"]


def plotdata(name, values):
    df = pd.DataFrame(values)

    # Remove outliers
    # https://stackoverflow.com/a/23202269
    # filtered = df
    filtered = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    # 1-value sets get 100% filtered
    if len(filtered) == 0:
        filtered = df

    total = len(df)
    removed = len(df) - len(filtered)
    print("[%s] Removed %d of %d (%d%%)" % (name, removed, total, 100*removed/total))
    name += " (%d%%)" % (100*len(filtered)/total)
    return name, filtered


modelfile = "data/fe-arch/l3-do-bn-fdim1024-rollouts100-lr0.0001-e0-1706598388/model.zip"
# modelfile = "data/fe-arch/l3-do-bn-fdim1024-1706303678/model.zip"

model = MaskablePPO.load(modelfile)
plots = []

for i, net in enumerate(model.policy.features_extractor.network):
    if not net.state_dict():
        continue
    name = "fe/%d.%s.%s" % (i, net.__class__.__name__, t)
    values = net.state_dict()[t].flatten()
    plots.append(plotdata(name, values))

if model.policy.action_net.state_dict():
    name = "action/%s" % t
    values = model.policy.action_net.state_dict()[t].flatten()
    plots.append(plotdata(name, values))

if model.policy.value_net.state_dict():
    name = "value/%s" % t
    values = model.policy.value_net.state_dict()[t].flatten()
    plots.append(plotdata(name, values))


# fig, axes = plt.subplots(nrows=2, ncols=len(plots)//2)
# fig, axes = plt.subplots(nrows=2, ncols=len(plots)//2, figsize=(15, 5))
plt.ticklabel_format(axis="x", useMathText=True)
fig = plt.figure(figsize=(15, 7))
n_cols = 3
n_rows = math.ceil(len(plots) / n_cols)

for i, (name, df) in enumerate(plots):
    # mask = (values >= VMIN) & (values <= VMAX)
    # filtered = values[mask]
    # total = len(values)
    # removed = len(values) - len(filtered)
    # print("[%s] Removed %d of %d (%d%%)" % (name, removed, total, 100*removed/total))
    # axes[i].hist(filtered, bins=50, color='blue', alpha=0.7)

    ax = fig.add_subplot(n_rows, n_cols, i+1)
    df[0].hist(bins=50, ax=ax, color='blue', alpha=0.7)
    # axes[i].hist(values, bins=50, color='blue', alpha=0.7)
    ax.set_title(name)
    # ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))


fig.tight_layout()
plt.show()
