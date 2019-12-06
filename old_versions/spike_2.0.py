"""
Creation date: 21/05/19
Last update: 23/07/19
Author: Federico Cascino Milani
Title: Spiking neuron net
Description:
"""

# ------------ IMPORT ------------ #

import numpy as np
import matplotlib.pyplot as plt
import pickle


# ------------ CLASSES ------------ #

# Simulator
class Sim:
    # Simulation parameters
    # Global settings
    time = 2000.                     # total time simulation in ms
    dt = 1.                          # timestep in ms
    duration = int(time / dt)        # steps during simulation
    save_data = False                # save simulation results

    # Net dimensions
    n_neurons = 10                   # n × n is number of output neurons
    m_inp_nodes = 10                 # m × m is number of input neurons
    input_w_range = (0., 1800.)      # weights range for W_X_OUT matrix

    # Lateral inibition (see beta in Net class)
    positive_sigma = 0.3             # sigma^(-1) of positive gaussian
    negative_sigma = 0.05            # sigma^(-1) of negative gaussian
    r = 0.1                          # gaussian traslation along y axis
    plot_beta = True                 # plot beta function

    # Neuron parameters
    abs_ref = int(5. / dt)           # absolute refractory time in ms
    v_rest = -70.                    # resting potential in mV
    v_reset = -85.                   # iperpolarisation after spike in mV
    v_spike = 40.                    # voltage during spike
    threshold = -35.                 # threshold for spike in mv
    tau = 4.                         # time costant in ms
    v_out = 1.                       # voltage variation in postinaptic neurons

    # Run simulation
    def startsim(self):
        self.inp = InputNet(Sim.m_inp_nodes, Sim.n_neurons)
        self.pc = Net(Sim.n_neurons)
        self.wr_data()
        self.inp.weight()
        print(self.inp.W_X_OUT)
        # print(self.pc.NETCON)
        for t in range(1, Sim.duration):
            self.inp.inp_activation(t)
            self.pc.activation(t, self.inp)
        self.pc.frequency(0, int(Sim.duration / 3))
        self.pc.frequency(int(Sim.duration * 2 / 3), Sim.duration, 1)
        self.inp.frequency()

    # Write on file
    def wr_data(self, flag=False):
        data = []
        if flag:
            with open('test.dat', 'rb') as f:
                data = pickle.load(f)
                a = np.array(data[0])
                b = np.array(data[1])
                self.inp.temp = a
                self.pc.W = b
        else:
            with open('test.dat', 'wb') as f:
                data.append(self.inp.temp)
                data.append(self.pc.W)
                pickle.dump(data, f)


# Neuron properties
class Neuron:
    # Data structures
    def __init__(self):
        # refractory counter (used in changev)
        self.t_ref = 0

        # neuron voltage during time (mV)
        self.voltage = np.zeros(Sim.duration)
        # cell starts at resting potential
        self.voltage[0] = Sim.v_rest

        # history of spikes
        self.fire = np.zeros(Sim.duration, dtype=int)
        # input current to neuron at time t (sum of other neurons activation)
        self.v_in = np.full(Sim.duration, Sim.v_rest)

    # Modify membrane potential if voltage > treshold
    # we have spike and ripolarisation
    def changev(self, t):
        if t > self.t_ref:
            # compute voltage by bacward euler derivative
            self.voltage[t] = self.voltage[t - 1] + \
                (Sim.dt / Sim.tau) * \
                (-self.voltage[t - 1] + self.v_in[t])

            # do spike and ripolarisation
            if self.voltage[t] >= Sim.threshold:
                self.voltage[t] = Sim.v_spike
                self.t_ref = t + Sim.abs_ref
                self.voltage[t + Sim.abs_ref: t: -1] = Sim.v_reset
                self.fire[t] = 1

            # set reset potential as min voltage value
            if self.voltage[t] < Sim.v_reset:
                self.voltage[t] = Sim.v_reset


class Net:
    # Generate net and weights, NETCON rapresent lateral inibition
    def __init__(self, n):
        # number of neurons
        self.n = n
        self.raster = []
        self.freq = np.zeros([self.n, self.n])
        self.freq1 = np.zeros([self.n, self.n])

        # distance matrix
        self.NETCON = np.zeros([self.n, self.n,
                                self.n, self.n])
        for i in range(self.n):
            for k in range(self.n):
                for j in range(self.n):
                    for l in range(self.n):
                        if i == j and k == l:
                            self.NETCON[i, k][j, l] = 0.
                        else:
                            self.NETCON[i, k][j, l] = self.beta(i, k, j, l)

        # lateral connection matrix
        self.W = np.random.rand(self.n, self.n,
                                self.n, self.n)
        for i in range(self.n):
            for k in range(self.n):
                self.W[i][k][i][k] = 0.

        # Generate net rapresention with object array
        self.neurons = np.zeros([self.n, self.n], dtype=object)
        for i in range(self.n):
            for k in range(self.n):
                self.neurons[i, k] = Neuron()

    # Interneuron distance factor
    def beta(self, xs, ys, x, y):
        max = Sim.n_neurons
        r = Sim.r
        s1 = Sim.positive_sigma
        s2 = Sim.negative_sigma
        dx = abs(xs - x)
        dy = abs(ys - y)
        if dx > (max / 2):
            dx = max - dx
        if dy > (max / 2):
            dy = max - dy
        d2 = dx * dx + dy * dy
        return np.exp(-d2 * s1) * (1 + r) - np.exp(-d2 * s2) * r

    # Distance decay function
    def gauss(self, x):
        return np.exp(-x * x)

    # Activation function
    # Check if neuron fired at t - 1 and compute his contribution to generate
    # PSP into neuron i,k at timestep t
    def activation(self, t, inp):
        # contribution of lateral neurons
        for i in range(self.n):
            for k in range(self.n):
                n1 = self.neurons[i, k]
                # for j in range(self.n):
                #     for l in range(self.n):
                #         n2 = self.neurons[j, l]
                #         if n2.fire[t - 1] == 1:
                #             n1.v_in[t] += self.NETCON[i, k][j, l]

                # summation of visual stimuli to neuron i,k
                for x in range(inp.m):
                    for y in range(inp.m):
                        m1 = inp.neurons[x, y]
                        if m1.fire[t - 1] == 1:
                            n1.v_in[t] += inp.W_X_OUT[x,
                                                      y][i, k] / (inp.m * inp.m)

                # change neuron voltage
                self.neurons[i, k].changev(t)

    # cal medium frequency and collect arrays for raster plot
    def frequency(self, a=0, b=Sim.duration, num=0):
        arrayinutile = np.arange(a, b, dtype=int)

        for i in range(self.n):
            for k in range(self.n):
                s = 0.
                r = []
                for t in arrayinutile:
                    s += self.neurons[i][k].fire[t]
                    if self.neurons[i][k].fire[t] == 1:
                        r.append(t)
                if num == 0:
                    self.freq[i, k] = s / (b - a)
                if num == 1:
                    self.freq1[i, k] = s / (b - a)
                self.raster.append(r)

    # def raster(self, t):
    #     self.r = np.zeros([self.n, self.n, Sim.duration])
    #     for i in range(self.n):
    #         for k in range(self.n):
    #             if self.neurons[i, k].fire[t] == 1:
    #                 self.r[i, k][t] = t


class InputNet(Net):
    def __init__(self, m=1, n=1):
        self.m = m
        self.n = n
        # Generate net rapresention with object array
        self.neurons = np.zeros([self.m, self.m], dtype=object)
        for i in range(self.m):
            for k in range(self.m):
                self.neurons[i, k] = Neuron()
        # visual stimuli during time
        self.X = np.zeros([self.m, self.m, Sim.duration])
        for t in range(int(Sim.duration / 3)):
            self.X[2, :, t] = 1.
        for t in range(int(Sim.duration * 2 / 3), Sim.duration):
            self.X[6, :, t] = 1.
        # self.visual()

        # weights from input to output
        self.temp = np.random.rand(m, m, n, n)

    def weight(self):
        a = Sim.input_w_range[0]
        b = Sim.input_w_range[1]
        self.W_X_OUT = (b - a) * self.temp + a
        # # scope matrix
        # self.XCON = np.zeros([self.m, self.m,
        #                       self.n, self.n])
        # self.scope()

    # # full input scope matrix
    # def scope(self):
    #     dn1 = int((self.n - self.m) / 2)
    #     for i in range(self.n):
    #         for k in range(self.n):
    #             for x in range(self.m):
    #                 for y in range(self.m):
    #                     self.XCON[x, y][i, k] = self.alfa(
    #                         x+dn1, y+dn1, i, k, 0.1, 0)

    # full visual input matrix
    def visual(self):
        v = int(1000 / 60)
        c = 0
        for t in range(Sim.duration):
            if c == 0:
                a = True
            if c == self.m - 1:
                a = False
            if a:
                self.X[:, c, t] = 1.
                if t % v == 0:
                    c += 1
            else:
                self.X[:, c, t] = 1.
                if t % v == 0:
                    c -= 1
            print(t)
            print(self.X[:, :, t])

    def inp_activation(self, t):
        for x in range(self.m):
            for y in range(self.m):
                m1 = self.neurons[x, y]
                if self.X[x, y][t - 1] == 1. and t > m1.t_ref:
                    m1.fire[t] = 1
                    m1.t_ref = t + Sim.abs_ref

    # # input distance decay function
    # def alfa(self, xs, ys, x, y, r, max):
    #     dx = abs(xs-x)
    #     dy = abs(ys-y)
    #     if dx > (max/2):
    #         dx = max-dx
    #     if dy > (max/2):
    #         dy = max-dy
    #     d2 = dx*dx + dy*dy
    #     g = np.exp(-d2 * 0.5) * (1 + r)-r
    #     if g > 0:
    #         return g
    #     else:
    #         return 0

    def frequency(self, a=0, b=Sim.duration, num=0):
        arrayinutile = np.arange(a, b, dtype=int)
        self.raster = []
        if num == 0:
            self.freq = np.zeros([self.m, self.m])
            f = self.freq
        if num == 1:
            self.freq1 = np.zeros([self.m, self.m])
            f = self.freq1
        for i in range(self.m):
            for k in range(self.m):
                s = 0.
                r = []
                for t in arrayinutile:
                    s += self.neurons[i][k].fire[t]
                    if self.neurons[i][k].fire[t] == 1:
                        r.append(t)
                f[i][k] = s / (a - b)
                self.raster.append(r)


# ------------ MAIN ------------ #
if __name__ == '__main__':
    S = Sim()
    S.startsim()


# ------------ PLOT ------------ #

    # for i in range(inp.n_neurons):
    #     for k in range(inp.n_neurons):
    #         print(pc.neurons[i][k].voltage)
    #         print(pc.neurons[i][k].w)

    # Raster plot
    col1 = np.random.rand(2 * (Sim.n_neurons ** 2), 3)
    col2 = np.random.rand((Sim.m_inp_nodes ** 2), 3)
    plt.figure(1)
    plt.subplot(211)
    plt.title('raster plot')
    plt.ylabel('output neurons')
    plt.eventplot(S.pc.raster, colors=col1)
    plt.subplot(212)
    plt.xlabel('time')
    plt.ylabel('input neurons')
    plt.eventplot(S.inp.raster, colors=col2)
    plt.show()

    # Frequency graphic
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

    # for i in range(pc.n_neurons):
    #     for k in range(pc.n_neurons):
    #         print(pc.neurons[i][k].freq)
    # fig, axs = plt.subplots(Sim.n_neurons, Sim.n_neurons)
    # for i in range(Sim.n_neurons):
    #     for k in range(Sim.n_neurons):
    #         axs[i, k].plot(S.pc.neurons[i][k].voltage)
    #         # print(inp.neurons[i][k].fire)
    # plt.show()

    # fig, axs = plt.subplots(Sim.m_inp_nodes, Sim.m_inp_nodes)
    # for i in range(Sim.m_inp_nodes):
    #     for k in range(Sim.m_inp_nodes):
    #         axs[i, k].plot(S.inp.neurons[i][k].voltage)
    #         # print(inp.neurons[i][k].fire)
    # plt.show()

    def plt_set(n, m, title):
        size = n * m * 0.08
        plt.ion()
        fig, axs = plt.subplots(n, n, sharex=True, sharey=True, gridspec_kw={
            'hspace': 0, 'wspace': 0}, figsize=(size, size))
        fig.suptitle(title)
        return axs

    def mat_show(axes, M):
        (xr, yr) = axes.shape
        for x in range(xr):
            for y in range(yr):
                axes[x, y].imshow(M[:, :][x, y], cmap=plt.cm.Blues)

    ax1 = plt_set(S.n_neurons, S.m_inp_nodes,
                  'Input weights by neuron position')
    mat_show(ax1, S.inp.W_X_OUT)
    plt.show()
    # plt.plot(S.pc.neurons[5][0].voltage)
    # plt.plot(S.pc.neurons[5][3].voltage)
    # plt.plot(S.pc.neurons[5][6].voltage)
    # plt.plot(S.pc.neurons[5][9].voltage)
    # plt.show()
