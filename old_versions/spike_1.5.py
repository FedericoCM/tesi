"""
Creation date: 21/05/19
Last update: 4/07/19
Author: Federico Cascino Milani
Title: Spiking neuron net
Description:
"""

# ------------ IMPORT ------------ #
import numpy as np
import matplotlib.pyplot as plt


# ------------ CLASSES ------------ #
# Simulation parameters
class Sim:
    time = 2000.                         # total time simulation in ms
    dt = 1.                            # timestep in ms
    duration = int(time / dt)          # steps during simulation

    # def startsim(self):
    #     self.pc = Net(2)
    #     self.pc.allcon()
    #     self.pc.gen()
    #     for t in range(Sim.duration):
    #         self.pc.activation(t)


# Neuron properties
class Neuron:
    abs_ref = int(5. / Sim.dt)      # absolute refractory time in ms
    v_rest = -70.                   # resting potential in mV
    v_reset = -85.                  # iperpolarisation after spike in mV
    v_spike = 40.                   # voltage during spike
    threshold = -35.                # threshold for spike in mv

    def __init__(self, pos, CON):
        self.pos = pos
        self.con = CON     # forse sta matrice di connessione andrebbe levata
        self.w = {}
        self.t_ref = 0                     # refractory counter
        self.v_out = 20.                    # postsinaptic potential
        self.v_in = np.full(Sim.duration, Neuron.v_rest)
        self.voltage = np.zeros(Sim.duration)      # neuron voltage (mv)
        self.voltage[0] = Neuron.v_rest    # cell starts at resting potential
        self.tau = 10.                     # time costant in ms
        self.fire = np.zeros(Sim.duration, dtype=int)  # history of spikes
        self.freq = 0.

        for i in range(self.con.shape[0]):
            for k in range(self.con.shape[1]):
                if self.con[i][k] != 0:
                    char = str(i) + ',' + str(k)
                    self.w[char] = 1.
                    # self.w[char] = 20. * np.random.random_sample() - 5.

    def input_w(self, m):
        self.xin = np.random.rand(m, m) * 5.

    # modify membrane potential if voltage > treshold
    # we have spike and ripolarisation
    def changev(self, t):
        if t > self.t_ref:
            self.voltage[t] = self.voltage[t - 1] + \
                (Sim.dt / self.tau) * \
                (-self.voltage[t - 1] + self.v_in[t])

            if self.voltage[t] >= Neuron.threshold:
                self.voltage[t] = Neuron.v_spike
                self.t_ref = t + Neuron.abs_ref
                self.voltage[t + Neuron.abs_ref: t: -1] = Neuron.v_reset
                self.fire[t] = 1

            if self.voltage[t] < Neuron.v_reset:
                self.voltage[t] = Neuron.v_reset

    def frequency(self):
        s = 0.
        interval = Sim.duration - 1000.
        for t in range(1000, Sim.duration):
            s += self.fire[t]
        self.freq = s / interval


class Net:
    def __init__(self, n=1):
        self.n_neurons = n
        self.CON = np.zeros((self.n_neurons, self.n_neurons,
                             self.n_neurons, self.n_neurons), dtype=int)
        # distance matrix
        self.NETCON = np.zeros([self.n_neurons, self.n_neurons,
                                self.n_neurons, self.n_neurons])
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                for j in range(self.n_neurons):
                    for l in range(self.n_neurons):
                        if i == j and k == l:
                            self.NETCON[j, l][i, k] = 0.
                        else:
                            self.NETCON[j, l][i, k] = self.beta(
                                j, l, i, k, 0.4, self.n_neurons)

    def allcon(self):
        self.CON = np.ones((self.n_neurons, self.n_neurons,
                            self.n_neurons, self.n_neurons), dtype=int)
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.CON[i][k][i][k] = 0.

    def gen(self, n=1, m=0):
        self.neurons = np.zeros((n, n), dtype=object)
        for i in range(n):
            for k in range(n):
                char = str(i) + ',' + str(k)
                self.neurons[i][k] = Neuron(char, self.CON[i, k, :, :])
                if m != 0:
                    self.neurons[i][k].input_w(m)

    # interneuron distance factor

    def beta(self, xs, ys, x, y, r, max):
        dx = abs(xs-x)
        dy = abs(ys-y)
        if dx > (max/2):
            dx = max-dx
        if dy > (max/2):
            dy = max-dy
        d2 = dx*dx + dy*dy
        return np.exp(-d2*0.6)*(1+r)-np.exp(-d2*0.001)*r

    # distance decay function
    def gauss(self, x):
        return np.exp(-x * x)

    def activation(self, t, net):
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                n1 = self.neurons[i][k]
                if n1.fire[t - 1] == 1:
                    for j in range(self.n_neurons):
                        for l in range(self.n_neurons):
                            n2 = self.neurons[j][l]
                            if n1.pos in \
                               n2.w.keys():
                                n2.v_in[t] += n1.v_out * \
                                    n2.w[n1.pos] * self.NETCON[i, k][j, l]

                s_inp = 0.         # summation of visual stimuli to neuron i,k
                for x in range(net.m_inp_nodes):
                    for y in range(net.m_inp_nodes):
                        m1 = net.neurons[x][y]
                        if m1.fire[t - 1] == 1:
                            s_inp += m1.v_out * n1.xin[x][y]
                n1.v_in[t] += s_inp

        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.neurons[i][k].changev(t)

    def net_freq(self):
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.neurons[i][k].frequency()


class InputNet(Net):
    def __init__(self, m=1, n=1):
        self.m_inp_nodes = m  # number of input nodes
        self.n_neurons = n    # number of net neurons
        self.CON = np.zeros((self.n_neurons, self.n_neurons,
                             self.n_neurons, self.n_neurons), dtype=int)
        self.X = np.zeros([self.m_inp_nodes, self.m_inp_nodes, Sim.duration])
        self.visual()         # visual stimuli during time
        # # scope matrix
        # self.XCON = np.zeros([self.m_inp_nodes, self.m_inp_nodes,
        #                       self.n_neurons, self.n_neurons])
        # self.scope()

    # # full input scope matrix
    # def scope(self):
    #     dn1 = int((self.n_neurons - self.m_inp_nodes) / 2)
    #     for i in range(self.n_neurons):
    #         for k in range(self.n_neurons):
    #             for x in range(self.m_inp_nodes):
    #                 for y in range(self.m_inp_nodes):
    #                     self.XCON[x, y][i, k] = self.alfa(
    #                         x+dn1, y+dn1, i, k, 0.1, 0)

    # full visual input matrix
    def visual(self):
        v = int(1000 / 60)
        c = 0
        for t in range(Sim.duration):
            if c == 0:
                a = True
            if c == self.m_inp_nodes - 1:
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
        for x in range(self.m_inp_nodes):
            for y in range(self.m_inp_nodes):
                m1 = self.neurons[x][y]
                if self.X[x, y][t - 1] == 1. and t > m1.t_ref:
                    m1.fire[t] = 1
                    m1.t_ref = t + Neuron.abs_ref

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


# ------------ MAIN ------------ #
S = Sim()
inp = InputNet(4, 4)
inp.gen(inp.m_inp_nodes)
pc = Net(4)
pc.allcon()
pc.gen(pc.n_neurons, inp.m_inp_nodes)
for t in range(1, Sim.duration):
    inp.inp_activation(t)
    pc.activation(t, inp)
pc.net_freq()

# ------------ PLOT ------------ #
# for i in range(inp.n_neurons):
#     for k in range(inp.n_neurons):
#         print(pc.neurons[i][k].voltage)
#         print(pc.neurons[i][k].w)

fig, axs = plt.subplots(inp.m_inp_nodes, inp.m_inp_nodes)
for i in range(inp.m_inp_nodes):
    for k in range(inp.m_inp_nodes):
        axs[i, k].plot(inp.neurons[i][k].fire)
        # print(inp.neurons[i][k].fire)

fig, axs = plt.subplots(pc.n_neurons, pc.n_neurons)
for i in range(pc.n_neurons):
    for k in range(pc.n_neurons):
        axs[i, k].plot(pc.neurons[i][k].fire)


for i in range(pc.n_neurons):
    for k in range(pc.n_neurons):
        print(pc.neurons[i][k].freq)

# axs[0, 0].plot(pc.neurons[5][0].voltage)
# axs[0, 1].plot(pc.neurons[5][3].voltage)
# axs[1, 0].plot(pc.neurons[5][6].voltage)
# axs[1, 1].plot(pc.neurons[5][9].voltage)
