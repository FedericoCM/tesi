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
import pickle


# ------------ CLASSES ------------ #

# Simulation parameters
class Sim:
    time = 2000.                       # total time simulation in ms
    dt = 1.                            # timestep in ms
    duration = int(time / dt)          # steps during simulation

    def startsim(self):
        self.inp = InputNet(4, 10)
        self.pc = Net(10)
        self.wr_data(True)
        # self.wr_data()
        self.inp.weight()
        print(self.inp.W_X_OUT)
        print(self.pc.NETCON)
        self.inp.gen(self.inp.m_inp_nodes)
        self.pc.gen(self.pc.n_neurons)
        for t in range(1, Sim.duration):
            self.inp.inp_activation(t)
            self.pc.activation(t, self.inp)
        self.pc.frequency(0, 1000)
        self.pc.frequency(1000, Sim.duration, 1)
        self.inp.frequency()

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

    def __init__(self):
        self.t_ref = 0                     # refractory counter
        self.v_out = 1.                   # postsinaptic potential
        self.v_in = np.full(Sim.duration, Neuron.v_rest)
        self.voltage = np.zeros(Sim.duration)      # neuron voltage (mv)
        self.voltage[0] = Neuron.v_rest    # cell starts at resting potential
        self.tau = 10.                     # time costant in ms
        self.fire = np.zeros(Sim.duration, dtype=int)  # history of spikes
        # self.freq = 0.

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


class Net:
    def __init__(self, n=1):
        self.n_neurons = n   # neurons number
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
                                j, l, i, k, 0.1, self.n_neurons)
        # intranet weight matrix
        self.W = np.random.rand(self.n_neurons, self.n_neurons,
                                self.n_neurons, self.n_neurons)
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.W[i][k][i][k] = 0.

    def gen(self, n=1):
        self.neurons = np.zeros((n, n), dtype=object)
        for i in range(n):
            for k in range(n):
                self.neurons[i][k] = Neuron()

    # interneuron distance factor
    def beta(self, xs, ys, x, y, r, max):
        dx = abs(xs-x)
        dy = abs(ys-y)
        if dx > (max/2):
            dx = max-dx
        if dy > (max/2):
            dy = max-dy
        d2 = dx*dx + dy*dy
        return np.exp(-d2*0.3)*(1+r)-np.exp(-d2*0.05)*r

    # distance decay function
    def gauss(self, x):
        return np.exp(-x * x)

    def activation(self, t, net):
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                n1 = self.neurons[i][k]
                # if n1.fire[t - 1] == 1:
                #     for j in range(self.n_neurons):
                #         for l in range(self.n_neurons):
                #             n2 = self.neurons[j][l]
                #             n2.v_in[t] += n1.v_out * \
                #                 self.NETCON[i, k][j, l]

                s_inp = 0.         # summation of visual stimuli to neuron i,k
                for x in range(net.m_inp_nodes):
                    for y in range(net.m_inp_nodes):
                        m1 = net.neurons[x][y]
                        if m1.fire[t - 1] == 1:
                            s_inp += m1.v_out * net.W_X_OUT[x, y][i, k]
                n1.v_in[t] += s_inp

        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.neurons[i][k].changev(t)

    def frequency(self, a=0, b=Sim.duration, num=0):
        arrayinutile = np.arange(a, b, dtype=int)
        self.raster = []
        if num == 0:
            self.freq = np.zeros([self.n_neurons, self.n_neurons])
            a = self.freq
        if num == 1:
            self.freq1 = np.zeros([self.n_neurons, self.n_neurons])
            a = self.freq1
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                s = 0.
                r = []
                for t in arrayinutile:
                    s += self.neurons[i][k].fire[t]
                    if self.neurons[i][k].fire[t] == 1:
                        r.append(t)
                    a[i][k] = s / Sim.duration
                self.raster.append(r)

    # def raster(self, t):
    #     self.r = np.zeros([self.n_neurons, self.n_neurons, Sim.duration])
    #     for i in range(self.n_neurons):
    #         for k in range(self.n_neurons):
    #             if self.neurons[i, k].fire[t] == 1:
    #                 self.r[i, k][t] = t


class InputNet(Net):
    def __init__(self, m=1, n=1):
        self.m_inp_nodes = m  # number of input nodes
        self.n_neurons = n    # number of net neurons

        # visual stimuli during time
        self.X = np.zeros([self.m_inp_nodes, self.m_inp_nodes, Sim.duration])
        for t in range(1000):
            self.X[2, :, t] = 1.
        for t in range(1000, Sim.duration):
            self.X[0, :, t] = 1.
        # self.visual()

        # weights from input to output
        self.temp = np.random.rand(m, m, n, n)

    def weight(self):
        omega = 60.
        self.W_X_OUT = self.temp * omega
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

    def frequency(self, a=0, b=Sim.duration, num=0):
        arrayinutile = np.arange(a, b, dtype=int)
        self.raster = []
        if num == 0:
            self.freq = np.zeros([self.m_inp_nodes, self.m_inp_nodes])
            a = self.freq
        if num == 1:
            self.freq1 = np.zeros([self.m_inp_nodes, self.m_inp_nodes])
            a = self.freq1
        for i in range(self.m_inp_nodes):
            for k in range(self.m_inp_nodes):
                s = 0.
                r = []
                for t in arrayinutile:
                    s += self.neurons[i][k].fire[t]
                    if self.neurons[i][k].fire[t] == 1:
                        r.append(t)
                    a[i][k] = s / Sim.duration
                self.raster.append(r)


# ------------ MAIN ------------ #

S = Sim()
S.startsim()


# ------------ PLOT ------------ #

# for i in range(inp.n_neurons):
#     for k in range(inp.n_neurons):
#         print(pc.neurons[i][k].voltage)
#         print(pc.neurons[i][k].w)


# raster plot
# fig, axs = plt.subplots(pc.n_neurons)
# for i in range(pc.n_neurons):
#     axs[i].eventplot(pc.raster[i, :])
col1 = np.random.rand(S.pc.n_neurons ** 2, 3)
col2 = np.random.rand(S.inp.m_inp_nodes ** 2, 3)
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

# plt.figure(2)
# plt.subplot(211)
# plt.title('spike frequency of neurons')
# plt.imshow(pc.freq)
# plt.subplot(212)
# plt.imshow(pc.freq1)
# plt.show()

fig, (ax, ax2) = plt.subplots(1, 2)
im = ax.imshow(S.pc.freq)
im2 = ax2.imshow(S.pc.freq1)

# We want to show all ticks...
ax.set_xticks(np.arange(S.pc.n_neurons))
ax.set_yticks(np.arange(S.pc.n_neurons))
ax2.set_xticks(np.arange(S.pc.n_neurons))
ax2.set_yticks(np.arange(S.pc.n_neurons))
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")
# plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(S.pc.n_neurons):
    for j in range(S.pc.n_neurons):
        text = ax.text(j, i, S.pc.freq[i, j],
                       ha="center", va="center", color="w")
        text2 = ax2.text(j, i, S.pc.freq1[i, j],
                         ha="center", va="center", color="w")

ax.set_title("Neurons spike frequency 0-1000")
ax2.set_title("Neurons spike frequency 1000-2000")
fig.tight_layout()
plt.show()
# for i in range(pc.n_neurons):
#     for k in range(pc.n_neurons):
#         print(pc.neurons[i][k].freq)
fig, axs = plt.subplots(S.pc.n_neurons, S.pc.n_neurons)
for i in range(S.pc.n_neurons):
    for k in range(S.pc.n_neurons):
        axs[i, k].plot(S.pc.neurons[i][k].voltage)
        # print(inp.neurons[i][k].fire)
plt.show()

# plt.plot(S.pc.neurons[5][0].voltage)
# plt.plot(S.pc.neurons[5][3].voltage)
# plt.plot(S.pc.neurons[5][6].voltage)
# plt.plot(S.pc.neurons[5][9].voltage)
# plt.show()
