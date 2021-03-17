import numpy as np

def softmax(x):
    return np.exp(x)/np.exp(x).sum()

def plot_strategy(data):
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(data=data, ax=ax, cbar_kws={"boundaries": np.linspace(-1, 2, 3)})
    ax.set_yticklabels(np.arange(4,22))
    ax.set_xticklabels(np.arange(4,22))
    ax.invert_yaxis()

    # colorbar modifications
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(['hit', 'stick'])

    plt.xlabel('Player Card Sum');
    plt.ylabel('Dealer Card Sum');
    plt.title("Greedy Strategy for each State");
