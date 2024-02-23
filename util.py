import numpy as np
import matplotlib.pyplot as plt


def plot_cum_reward(ax, reward_dict, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(reward_dict) - 1))

    for i, (key, value) in enumerate(reward_dict.items()):
        if i == 0:
            color = 'black'
        else:
            color = colors[i-1]
        ax.plot(np.cumsum(value), label=key, color=color)
            
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Cummulative Reward')
    ax.set_xlabel('Round')
    return ax


def plot_budget(ax, meta, title):
    colors = plt.cm.tab10(np.linspace(0, 1, len(meta['bank_budget_dict'])))
    ax.plot(meta['algo_budget_list'], label='algo', color='black')
    for i, (key, value) in enumerate(meta['bank_budget_dict'].items()):
        ax.plot(value, label=key, color=colors[i])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel('Remaining Budget')
    return ax


def get_running_mean(data, window_size):
    """
    Calculate the running mean of a list of values using NumPy.

    Parameters:
    data (list or numpy.ndarray): The input data array.
    window_size (int): The size of the window for the running mean.

    Returns:
    numpy.ndarray: The array of running mean values.
    """
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, 'valid')


def plot_bid(ax, meta, title, window_size=5):
    colors = plt.cm.tab10(np.linspace(0, 1, len(meta['bank_budget_dict'])))
    for i, (grade, values) in enumerate(dict(sorted(meta['algo_bid_dict'].items())).items()):
        running_mean = get_running_mean(values, window_size)
        ax.plot(running_mean, color=colors[i], label=f'GRADE: {grade}')
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel('Bid')
    return ax
    
