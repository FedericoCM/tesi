"""
This class takes data from net and print them on screen
"""


# ------------ IMPORT ------------ #
import matplotlib.pyplot as plt
import numpy as np

# ------------ PLOTTER ------------ #


class Plotter:
    def __init__(self, duration, dimensions, n_neurons):
        self.duration = duration
        self.dimensions = dimensions
        self.n_neurons = n_neurons

    def raster_plot(self, data):
        # Raster plot
        rows = len(self.dimensions)
        np.random.seed(34)
        colo = [np.random.rand(self.n_neurons[i], 3)
                for i in range(rows)]

        fig, axes = plt.subplots(nrows=rows, ncols=1)
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle('Raster Plot')

        for ax, index, layer in zip(axes.flatten(), range(rows), data):
            if len(layer) != 0:
                ax.eventplot(layer, colors=colo[index])
            else:
                ax.eventplot(layer)
            ax.set_ylabel('neurons layer_'+str(index), fontsize=8)
            if index == rows - 1:
                ax.set_xlabel('time', fontsize=10)

        plt.show()

    def volt_plot(self, v_in, voltage):
        rows = len(self.dimensions)
        columns = max(self.n_neurons)
        height = np.ones_like(self.n_neurons)
        widht = [(columns / n) for n in self.n_neurons]
        gs_kw = dict(widht_ratios=widht, height_ratios=height)

        fig, axes = plt.subplots(ncols=columns, nrows=rows,
                                 constrained_layout=True, gridspec_kw=gs_kw)
        for row in axes:
            for ax in row:
                ax.plot(v_in)
                ax.plot(voltage)

    def frequency_graph(self):
        fig, (ax, ax2) = plt.subplots(1, 2)
        im = ax.imshow(S.pc.freq)
        im2 = ax2.imshow(S.pc.freq1)

        # Show all ticks...
        ax.set_xticks(np.arange(Sim.n_neurons))
        ax.set_yticks(np.arange(Sim.n_neurons))
        ax2.set_xticks(np.arange(Sim.n_neurons))
        ax2.set_yticks(np.arange(Sim.n_neurons))

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")
        # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(Sim.n_neurons):
            for j in range(Sim.n_neurons):
                text = ax.text(j, i, '{:0.4f}'.format(S.pc.freq[i, j]),
                               ha="center", va="center", color="w")
                text2 = ax2.text(j, i, '{:0.4f}'.format(S.pc.freq1[i, j]),
                                 ha="center", va="center", color="w")

        ax.set_title("Neurons spike frequency 0-1000")
        ax2.set_title("Neurons spike frequency 1000-2000")
        fig.tight_layout()
        plt.show()


# dimensions = [(2, 2), (2, 1), (1, 1)]
# n_neurons = [4, 2, 1]
# duration = 2

# p = Plotter(duration, dimensions, n_neurons)

# a = []

# for i in n_neurons:
#     b = np.random.rand(i, duration)
#     a.append(b)

# a = np.array(a)
# c = a * 10
# print(a, c)
# p.volt_plot(a, c)
