"""
This class defines net size and its connections. Calculates neurons
activation and saves net state
"""
# FARE IL DEBUG PER VEDERE SE FUNZIONA IL LEAKY
# INPUT
# PLOTTER
# GAUSSIANE


# ------------ IMPORT ------------ #
import numpy as np
import matplotlib.pyplot as plt
import neuron as nrn


# ------------ NET ------------ #
class Net:
    """ Net is the core of simulation. Every instance takes care of
    generate neurons, calculate input currents and neurons activation,
    modify weights and save data. Net could be execute in two different
    modality: spike and leaky. In spike mode fire frequency is important.
    In leaky mode neurons don't fire and is considered only voltage variation
    """

    # DEFAULT PARAMETERS #
    """ Here default values are setted. Usually these parameters
    are modified during instance creation because they change according
    to simulation """
    # total time of simulation in timesteps
    duration = 500

    # net dimensions
    """ Layer's dimensions expressed by (x,y) value, x is the width
    and y the height, layer number 0 points out input neurons """
    dimensions = [(2, 2), (2, 1), (1, 1)]

    # spike/leaky options
    """ In spike mode, voltage variation on post-sinaptic neuron
    should be 10-100 times higher than leaky mode, in order to produce
    significant depolarisation for spike. For this reason, in spike mode,
    each input is multiplied by omicron factor during net activity """
    spike = True                # I&F flag
    omicron = 100.              # input multiplication factor

    def __init__(self, **attr):
        """ This class takes the following parameters:
        (duration, dimensions, spike, omicron) """
        # PARAMETERS #
        self.__dict__.update(attr)  # update attributes values

        # number of layers
        self.layers = np.arange(len(self.dimensions))
        # neurons number for each layer
        self.n_neurons = [i * k for i, k in self.dimensions]

        # create 2D coordinates map based on dimensions
        """ Neuron objects are stored in linear array, to reproduce 2D
        every neuron has l,i,k coordinates where l is the layer, i is row,
        and k is column """
        self.coordinates_map = []
        self.name_map = []
        for layer in self.layers:  # da cambiare
            width = self.dimensions[layer][0]
            height = self.dimensions[layer][1]
            coordinates = [(layer, i, k) for i in range(width)
                           for k in range(height)]
            name = [(layer, n) for n in range(self.n_neurons[layer])]
            self.coordinates_map.append(coordinates)
            self.name_map.append(name)

        # DATA STRUCTURES #
        self.grid = []          # contains neurons

        """ In spike mode voltage variation is calculated for one neuron at time.
        Index of firing neurons are stored and their output is spread to others
        In leaky mode matrix calculus is used to compute input/output """
        # spike attributes
        if self.spike:
            self.firing = []    # register of firing neurons

    # GRID GENERATION FUNCTION #
    def grid_gen(self, layer, **attr):
        """ The grid is l-dimensional list that contains neuron instances,
        where l is layer number. Strata are created one by one based on user's
        specifications: layer level and neurons attributes(dictionary form) """
        stratum = []
        # creates layer
        for neuron in range(self.n_neurons[layer]):
            n = nrn.Neuron(duration=self.duration, name=(layer, neuron),
                           coordinates=(self.coordinates_map[layer][neuron]),
                           **attr)
            stratum.append(n)
        # append layer
        self.grid.append(stratum)

    # CONNECTIONS AND WEIGHTS FUNCTIONS #
    def all2all(self, first_layer, second_layer):
        """ Generate m×n weight matrix with random values from 0-1.
        Pass connections to neurons of the first layer """
        # set random weights
        w = np.random.rand(self.n_neurons[first_layer],
                           self.n_neurons[second_layer])
        # each neuron of the first layer receives second layer coordinates
        for index, neuron in enumerate(self.grid[first_layer]):
            neuron.add_connection(dict(zip(self.name_map[second_layer],
                                           w[index, :])))

    def intraconnection(self, layer, recurrent=False):
        """ Generate n×n weight matrix for intralayer connections. Random
        values from 0-1. if recurrent=True neurons are self-connected """
        w = np.random.rand(self.n_neurons[layer], self.n_neurons[layer])
        if recurrent is False:
            np.fill_diagonal(w, 0.)
        # neuron receives only weights != 0
        for index, neuron in enumerate(self.grid[layer]):
            connection = dict([(name, weight) for name, weight in zip(
                self.name_map[layer], w[index, :]) if weight != 0.])
            neuron.add_connection(connection)

    # ACTIVATION FUNCTIONS #
    def synapse(self, t, spike=True):
        """ Takes connections and weights of firing neuron and
        sets up input voltage into boundary neurons. In leaky mode takes
        voltage from layer at time t-1 to compute voltage at time t """
        # spike mode
        if self.spike:
            # iterates firing register and takes connections
            for layer, neuron in self.firing:
                connections = self.grid[layer][neuron].connections
                # iterates connections and pass voltage to boundary neurons
                for term, w in connections.items():
                    voltage = self.grid[layer][neuron].v_out * w * self.omicron
                    self.grid[term[0]][term[1]].set_v_in(t, voltage)
        # leaky mode
        else:
            # iterates neuron and takes connections
            for layer in self.grid:
                for neuron in layer:
                    connections = neuron.connections
                    # in leaky mode voltage is normalised at 0 whith v_rest
                    for term, w in connections.items():
                        voltage = neuron.voltage[t - 1] - neuron.v_rest
                        # if voltage is <0 weights are inverted
                        if voltage >= 0.:
                            voltage = (neuron.v_out * w *
                                       voltage) + neuron.v_rest
                        else:
                            voltage = ((-1 * neuron.v_out * w)
                                       * voltage) + neuron.v_rest
                        self.grid[term[0]][term[1]].set_v_in(t, voltage)

    def activation(self):
        """ Key function, used to calculate input currents at each timestep """
        # spike mode
        if self.spike:
            for t in range(1, self.duration):
                self.synapse(t)
                self.firing.clear()
                for layer in self.grid:
                    self.firing = list(
                        filter(None, map(lambda x: x.change_v(t), layer)))
        # leaky mode
        else:
            for t in range(1, self.duration):
                self.synapse(t)
                for layer in self.grid:
                    map(lambda x: x.change_v(t, self.spike), layer)

        #     # distance matrix, NETCON rapresent lateral inibition
        #     self.NETCON = np.zeros([self.n_neurons, self.n_neurons,
        #                             self.n_neurons, self.n_neurons])
        #     for i in range(self.n_neurons):
        #         for k in range(self.n_neurons):
        #             for j in range(self.n_neurons):
        #                 for l in range(self.n_neurons):
        #                     if i == j and k == l:
        #                         self.NETCON[i, k][j, l] = 0.
        #                     else:
        #                         self.NETCON[i, k][j, l] = self.beta(i, k, j, l)

        #     # lateral connection matrix
        #     self.W = np.random.rand(self.n_neurons, self.n_neurons,
        #                             self.n_neurons, self.n_neurons)
        #     for i in range(self.n_neurons):
        #         for k in range(self.n_neurons):
        #             self.W[i][k][i][k] = 0.

        # # Interneuron distance factor
        # def beta(self, xs, ys, x, y):
        #     max = Sim.n_neurons
        #     r = Sim.r
        #     s1 = Sim.positive_sigma
        #     s2 = Sim.negative_sigma
        #     dx = abs(xs - x)
        #     dy = abs(ys - y)
        #     if dx > (max / 2):
        #         dx = max - dx
        #     if dy > (max / 2):
        #         dy = max - dy
        #     d2 = dx * dx + dy * dy
        #     return np.exp(-d2 * s1) * (1 + r) - np.exp(-d2 * s2) * r

        # # Distance decay function
        # def gauss(self, x):
        #     return np.exp(-x * x)

        # # Calulate medium frequency and collect arrays for raster plot
        # def frequency(self, a=0, b=Sim.duration, num=0):
        #     arrayinutile = np.arange(a, b, dtype=int)

        #     for i in range(self.n):
        #         for k in range(self.n):
        #             s = 0.
        #             r = []
        #             for t in arrayinutile:
        #                 s += self.grid[i][k].fire[t]
        #                 if self.grid[i][k].fire[t] == 1:
        #                     r.append(t)
        #             if num == 0:
        #                 self.freq[i, k] = s / (b - a)
        #             if num == 1:
        #                 self.freq1[i, k] = s / (b - a)
        #             self.raster.append(r)

        # # def raster(self, t):
        # #     self.r = np.zeros([self.n, self.n, Sim.duration])
        # #     for i in range(self.n):
        # #         for k in range(self.n):
        # #             if self.grid[i, k].fire[t] == 1:
        # #                 self.r[i, k][t] = t


# ------------ MAIN (TEST) ------------ #
rete = Net()
leaky = Net(spike=False)
for l in range(3):
    rete.grid_gen(l)
    leaky.grid_gen(l)

rete.all2all(0, 1)
leaky.all2all(0, 1)
rete.intraconnection(1)
leaky.intraconnection(1)
rete.all2all(1, 2)
leaky.all2all(1, 2)
rete.all2all(2, 1)
leaky.all2all(2, 1)
rete.firing = [(0, 1), (1, 1)]
rete.activation()
leaky.activation()
