import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable


def plot_premium_bar(data, x='remained_time', y='remained_shares', name='1', max_time=5, max_share=30):
    """
    data: DataFrame, the data used to plot
    x: string, choose one feature as the x axis
    y: string, choose one feature as the y axis
    name: string, name label of the plot
    max_time: int, the maximum remaining time
    max_share: int, the maximum remaining share
    """

    data = data[data.remained_time != 0]
    data_to_plot = data.groupby([x, y], as_index=False).agg({'premium': ['mean']})
    data_to_plot.columns = [''.join(x) for x in data_to_plot.columns.ravel()]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x_data = np.array(data_to_plot[x])
    y_data = np.array(data_to_plot[y])
    z_data = np.zeros(len(x_data))
    dx = 0.5 * np.ones(len(x_data))
    dy = 0.5 * np.ones(len(y_data))
    dz = np.array(data_to_plot.premiummean)

    ax.bar3d(x_data, y_data, z_data, dx, dy, dz, color='grey')
    ax.set_xlim(max_time + 1, 1)
    ax.set_ylim(max_share + 0.5, 0.5)
    ax.set_xticks(np.arange(1.25, max_time + 1.25))
    ax.set_xticklabels(list(map(str, range(1, max_time + 1))))
    ax.set_xlabel('Remaining Timesteps', fontsize=18)
    ax.set_ylabel('Remaining Shares', fontsize=18)
    ax.set_zlabel('Action', fontsize=18)

    plt.show()
    fig.savefig('premium_bar_plot_' + name + '.png', dpi=200)
    plt.close(fig)

def plot_premium_ts(data, name='1', max_time=5, max_share=30):
    """
    data: DataFrame, the data used to plot
    name: string, name label of the plot
    max_time: int, the maximum remaining time
    max_share: int, the maximum remaining share
    """

    trunc = list(data[data.remained_time == float(max_time)].index)
    xx = np.arange(0, len(data) + len(trunc))
    yy = np.array(data.premium) + 0.01
    data_color = np.array(data.remained_shares) / max_share
    count = 0
    tick_pos = []
    for i in trunc:
        yy = np.insert(yy, i + count, 0)
        data_color = np.insert(data_color, i + count, 0)
        tick_pos.append(i + count)
        count += 1

    # print(tick_pos)
    tick_pos = [i for i in xx if i not in tick_pos]
    # print(tick_pos)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    my_cmap = plt.cm.get_cmap('GnBu')
    colors = my_cmap(data_color)

    ax.bar(xx, yy, color=colors)

    sm = ScalarMappable(cmap=my_cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm)
    cbar.set_label('Remaining Shares', rotation=270, labelpad=25)
    cbar.set_ticks(np.arange(0, 1.01, 1 / (max_share - 1)))
    cbar.set_ticklabels(np.arange(1, max_share + 1))
    ax.set_xlabel('Time Step (remaining time)')
    ax.set_ylabel('Premium')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(data.remained_time.map(int))
    ax.set_yticklabels([str(i)[0:4] for i in (ax.get_yticks()-0.01)])
    ax.set_ylim(0.005, ax.get_ylim()[1])

    plt.show()
    fig.savefig('premium_ts_plot_' + name + '.png', dpi=200)
    plt.close(fig)

#-----------------------------------------------------------------------------------------------------------------------------------------
data = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/strat_one_asset_fast/2019-04-21_15-30-20/iteration_info/itr_1.csv')
for i in range(2, 10):
    data1 = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/strat_one_asset_fast/2019-04-21_15-30-20/iteration_info/itr_' + str(i) + '.csv')
    data = pd.concat([data, data1], ignore_index=True)
plot_premium_bar(data,max_time=5, max_share=6)

data.to_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/strat_one_asset_fast/2019-04-21_15-30-20/iteration_info/sum.csv')

data = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/exec_one_asset/2019-04-04_13-04-38/iteration_info/itr_25.csv')
plot_premium_ts(data,max_time=5, max_share=6)


dt = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/result.csv')

#-------------------------------------------------------------------------------------------------
d = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/exec_one_asset/2019-03-13_11-14-05/Reward time series_Execution.csv')
d2 = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/exec_one_asset/2019-03-13_11-14-05/Reward time series.csv')
d3 = pd.read_csv('/mnt/c/Users/liuyu/PycharmProjects/FE800/RL/data/exec_one_asset/2019-03-13_11-14-05/a/Reward time series_Execution.csv')
#%%
d.head()
#%%
dd = d.Reward.cumsum().plot()
dd2 = d2.Reward.cumsum().plot()
dd3 = d3.Reward.cumsum().plot()

plt.gca().legend(('Trained Agent','Market Order','One Shot'))
plt.show()