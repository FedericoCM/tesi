import numpy as np
import matplotlib.pyplot as plt
import Neuron as nrn


class Net:
    # Generate net and weights, NETCON rapresent lateral inibition
    def __init__(self, n):
        # number of neurons
        self.n_neurons = n
        self.raster = []
        self.freq = np.zeros([self.n_neurons, self.n_neurons])
        self.freq1 = np.zeros([self.n_neurons, self.n_neurons])

        # distance matrix
        self.NETCON = np.zeros([self.n_neurons, self.n_neurons,
                                self.n_neurons, self.n_neurons])
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                for j in range(self.n_neurons):
                    for l in range(self.n_neurons):
                        if i == j and k == l:
                            self.NETCON[i, k][j, l] = 0.
                        else:
                            self.NETCON[i, k][j, l] = self.beta(i, k, j, l)

        # lateral connection matrix
        self.W = np.random.rand(self.n_neurons, self.n_neurons,
                                self.n_neurons, self.n_neurons)
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.W[i][k][i][k] = 0.

        # Generate net rapresention with object array
        self.grid = np.zeros([self.n_neurons, self.n_neurons], dtype=object)
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                self.grid[i, k] = nrn.Neuron()

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
        for i in range(self.n_neurons):
            for k in range(self.n_neurons):
                n1 = self.grid[i, k]
                # for j in range(self.n):
                #     for l in range(self.n):
                #         n2 = self.grid[j, l]
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
                self.grid[i, k].changev(t)

        # vol = 0.
        # p = []
        # o = []

        # # scan firing neurons
        # for i in range(self.n):
        #     for k in range(self.n):
        #         if self.grid[i, k].fire[t - 1] == 1:
        #             p.append((i, k))
        # for x in range(inp.m):
        #     for y in range(inp.m):
        #         if inp.neurons[x, y].fire[t - 1] == 1:
        #             o.append((x, y))

        # # sum input to neuron i,k
        # for i in range(self.n):
        #     for k in range(self.n):
        #         # sum input of firing neurons
        #         # for r in p:
        #         #     vol += self.NETCON[i, k][r]
        #         for s in o:
        #             vol += inp.W_X_OUT[s][i, k] / (inp.m * inp.m)

        #         # call change voltage function
        #         self.grid[i, k].changev(t, vol)
        # print(p, o)

    # Calulate medium frequency and collect arrays for raster plot
    def frequency(self, a=0, b=Sim.duration, num=0):
        arrayinutile = np.arange(a, b, dtype=int)

        for i in range(self.n):
            for k in range(self.n):
                s = 0.
                r = []
                for t in arrayinutile:
                    s += self.grid[i][k].fire[t]
                    if self.grid[i][k].fire[t] == 1:
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
    #             if self.grid[i, k].fire[t] == 1:
    #                 self.r[i, k][t] = t


class InputNet(Net):
    def __init__(self, m=1, n=1):
        self.m = m
        self.n = n
        # Generate net rapresention with object array
        self.grid = np.zeros([self.m, self.m], dtype=object)
        for i in range(self.m):
            for k in range(self.m):
                self.grid[i, k] = Neuron()
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
                m1 = self.grid[x, y]
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
                    s += self.grid[i][k].fire[t]
                    if self.grid[i][k].fire[t] == 1:
                        r.append(t)
                f[i][k] = s / (a - b)
                self.raster.append(r)
