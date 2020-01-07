
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

def plot_learning_curve_(xs, ys):
    sns.set(style="ticks", font_scale=1.1)
    fig, ax = plt.subplots()
    ax.plot(xs, ys)

    ax.set(xlabel="episode", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    plt.show()
    #fig.savefig(filename, format="pdf")
    plt.clf()



def plot_learning_curve(mse_dict, filename, label_fun, **kwargs):
    sns.set(style="ticks", font_scale=1.1)
    fig, ax = plt.subplots()
    i = 0
    for gamma, series in mse_dict.items():
        _label = label_fun(gamma, **kwargs) #label.format(gamma)
        ax.plot(series['x'], series['y'], label=_label)
        i += 1

    ax.set(xlabel="episode", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, format="pdf")
    plt.clf()


def plot_V(V, dealer_range, player_range, max_axis=2, save=None, fig_idx=0, init_view = (30,150)):
    def create_surf_plot(X, Y, Z, fig_idx=1):
        fig = plt.figure(fig_idx)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.view_init(init_view[0], init_view[1])
        return surf

    X, Y = np.mgrid[range(player_range), range(dealer_range)]
    surf = create_surf_plot(X, Y, V)
    plt.title("V*")
    plt.xlabel('player sum', size=18)
    plt.ylabel('dealer', size=18)

    if save is not None:
        plt.savefig(save, format='pdf', transparent=True)
    else:
        plt.show()
    plt.clf()    
