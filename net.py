"""
This class defines net size and its connections. Calculates neurons
activation and saves net state
"""


# ------------ IMPORT ------------ #
import numpy as np
import neuron as nrn
import retina as ret
import plotter as pl


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
    duration = 500             # total time of simulation in timesteps

    # net dimensions
    """ Layer's dimensions expressed by (x,y) value, x is the width
    and y the height, layer index 0 points out input neurons """
    dimensions = [(2, 2), (2, 1), (1, 1)]

    # spike/leaky options
    """ In spike mode, voltage variation on post-sinaptic neuron
    should be 10-100 times higher than leaky mode, in order to produce
    significant depolarisation for spike. For this reason, in spike mode,
    each input is multiplied by omicron factor during net activity """
    spike = True                # I&F flag
    omicron = 1000.              # input multiplication factor

    # visual input options
    """ Net inputs are images processed by retina class. Retina scan
    images at fixed frequency, so inputs arrive at fixed time intervals.
    In spike mode, input neurons respond with appropriate firing frequency
    respecting refractory time, for this reason it's passed like parameter """
    refresh_rate = 0.06         # frequency in mHz
    input_pulse = 100.          # PSP on input neurons in mV

    def __init__(self, **attr):
        """ This class takes the following parameters:
        (duration, dimensions, omicron, refresh_rate, spike) """
        # PARAMETERS #
        self.__dict__.update(attr)  # updates attributes values

        # neurons number for each layer
        self.n_neurons = [i * k for i, k in self.dimensions]

        # create 2D coordinates map based on dimensions
        """ Neuron objects are stored in linear array and each of them has
        index stored in name_map list. To reproduce 2D every neurons have
        l, i, k coordinates (where l is the layer, i is row, and k is column)
        saved in coordinates_map list """
        self.coordinates_map = []
        self.name_map = []
        # generates coordinates_map
        for layer, dim in enumerate(self.dimensions):
            coordinates = [(layer, i, k) for i in range(dim[0])
                           for k in range(dim[1])]
            self.coordinates_map.append(coordinates)
        # generates name_map
        for layer, lenght in enumerate(self.n_neurons):
            names = [(layer, n) for n in range(lenght)]
            self.name_map.append(names)

        # attributes for visual inputs
        # time interval between images in ms
        self.reload_period = 1 / self.refresh_rate

        # DATA STRUCTURES #
        self.grid = []          # contains neurons

        # creates retina object
        self.eye = ret.Retina(self.dimensions[0])  # process visual inputs

        # creates plotter object
        self.printer = pl.Plotter(
            self.duration, self.dimensions, self.n_neurons)  # plots data

        # spike attributes
        """ In spike mode voltage variation is calculated for one neuron at time.
        Index of firing neurons are stored and their output is spread to others
        """
        if self.spike:
            self.firing = []    # register of firing neurons

        # ATTRIBUTES FOR PLOTTER #
        self.rasplot = []
        self.M_freq = []

    # NET GENERATION FUNCTION #
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
    def all2all(self, first_layer, second_layer, rand=True, **attrs):
        """ Generate m×n weight matrix with random values from 0-1.
        Pass connections to neurons of the first layer. Finally, weights
        are passed to neuron and stored in a dictionary"""
        # set random weights
        if rand:
            w = np.random.rand(self.n_neurons[first_layer],
                               self.n_neurons[second_layer])
        # use beta to generate weights
        else:
            w = self.beta(layer, **attrs)
        # each neuron of the first layer receives second layer coordinates
        for index, neuron in enumerate(self.grid[first_layer]):
            neuron.add_connection(dict(zip(self.name_map[second_layer],
                                           w[index, :])))

    def intraconnection(self, layer, recurrent=False, rand=True, **attrs):
        """ Generate n×n weight matrix for intralayer connections. Random
        values from 0-1. if recurrent = True neurons are self-connected.
        Finally, weights are passed to neuron and stored in a dictionary """
        # random weights
        if rand:
            w = np.random.rand(self.n_neurons[layer], self.n_neurons[layer])
        # uses beta to generate weights
        else:
            w = self.beta(layer, **attrs)
        # if recurrent=False weight with itself is 0
        if recurrent is False:
            np.fill_diagonal(w, 0.)
        # neuron receives only weights != 0
        for index, neuron in enumerate(self.grid[layer]):
            connection = dict([(name, weight) for name, weight in zip(
                self.name_map[layer], w[index, :]) if weight != 0.])
            neuron.add_connection(connection)

    def beta(self, layer, **attrs):
        """ This function is used to generate weights that change with distance
        according to gaussian distrubution. x0, y0 are 2D coordinates of
        reference neuron; x1, y1 represent neuron to calculate distance """

        # side lenght, max distance between neurons
        max_distancex = self.dimensions[layer][0]
        max_distancey = self.dimensions[layer][1]

        A = attrs.get('A')                  # high(amplitude) of function
        sigma = attrs.get('sigma')          # standard deviation

        # distance between two points
        x_distance = abs(n1[0] - n0[0])
        y_distance = abs(n1[1] - n0[1])

        """ Makes net a closed surface, if distance is too long, count restarts
        from the opposite side """
        if x_distance > (max_distancex / 2):
            x_distance = max_distancex - x_distance
        if y_distance > (max_distancey / 2):
            y_distance = max_distancey - y_distance

        # square distance between neurons
        d2 = (x_distance * x_distance) + (y_distance * y_distance)

        # gaussian function
        return np.exp(-d2 * sigma) * A

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
                    # in leaky mode voltage is normalised with v_rest=0
                    for term, w in connections.items():
                        voltage = neuron.voltage[t - 1] - neuron.v_rest
                        # if voltage is <0 weights are inverted
                        if voltage >= 0.:
                            voltage = (neuron.v_out * w *
                                       voltage) + neuron.v_rest
                        else:
                            voltage = ((-1 * neuron.v_out * w)
                                       * voltage) + neuron.v_rest
                        # updates v_in
                        self.grid[term[0]][term[1]].set_v_in(t, voltage)

    def V1(self, inp, t):
        """ Input neurons have different chances to fire based on input array.
        If score(between 0-1) < chance this neuron receives one pulse """
        data = inp.flatten()
        for neuron, chance in enumerate(data):
            score = np.random.rand()
            if score < chance:
                self.grid[0][neuron].set_v_in(t, self.input_pulse)

    def activation(self):
        """ Key function, used to calculate input currents at each timestep """
        refresh = 0
        inp = []
        # spike mode
        if self.spike:
            for t in range(1, self.duration):
                if t > refresh:
                    inp = self.eye.vision()
                    self.collect_mfreq(t)
                    refresh = t + self.reload_period
                self.V1(inp, t)
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

    # FUNCTIONS TO COLLECT DATA #
    def collect_raster(self):
        for layer in self.grid:
            self.rasplot.append([neuron.raster() for neuron in layer])

    def collect_mfreq(self, t):
        temp = []
        for layer in self.grid:
            temp.append([neuron.mfreq(t, int(self.reload_period))
                         for neuron in layer])
        self.M_freq.append(temp)
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
if __name__ == '__main__':
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
    rete.collect_raster()
    print(rete.rasplot)
    rete.printer.raster_plot(rete.rasplot)
