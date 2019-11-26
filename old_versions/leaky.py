
"""
Prima prova per un modello leaky neurons
"""
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# functions
def beta(x, y):                 # distance decay function
    return np.exp(-(x-y) * (x-y))


def tanh(x):                    # tanh function
    if x <= 0:
        r = 0
    else:
        r = np.tanh(x)
    return r


# parameters and variables
T = 50.                          # total time simulation (ms)
dt = 0.2                         # timestep (ms)
time = np.arange(0, T+dt, dt)    # time array
n_neu = 10                       # neuron numbers
m_inp_nodes = 3                  # input nodes number
lt = len(time)                   # time array lenght

# pdb.set_trace()
# data matrices and initial settings
# visual stimuli in input X (random values [0, 1))
VI_X = np.random.rand(m_inp_nodes, m_inp_nodes)
# input weight matrix W (random values [0, 1))
INP_W = np.random.rand(n_neu, n_neu, m_inp_nodes, m_inp_nodes)
# neuron connection weight matrix Z
NETCON = np.ones([n_neu, n_neu, n_neu, n_neu])

for x in range(n_neu):           # set initial weights of Z
    for y in range(n_neu):
        for i in range(n_neu):
            for k in range(n_neu):
                if x == y == i == k:
                    NETCON[x][y][i][k] = 0.
                else:
                    NETCON[x][y][i][k] = 1.

# voltage matrix during time
NETV = np.zeros([n_neu, n_neu, lt])


# neurons properties
Rm = 1.                          # membrane resistance (kOhms)
Cm = 10.                         # membrane capacity (uF)
tau = Rm * Cm                    # time costant (ms)
I_in = np.zeros([n_neu, n_neu])    # current input for single neuron (nA)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# iteration
for t in range(1, lt):           # timestep dt
    for x in range(n_neu):       # cicle for neuron xy
        for y in range(n_neu):
            s_inp = 0.           # summation of visual stimuli coming to neuron xy
            for a in range(m_inp_nodes):
                for b in range(m_inp_nodes):
                    s_inp += VI_X[a][b] * INP_W[x][y][a][b]

            s_neu = 0.           # summation of neuron connection weights
            for i in range(n_neu):
                for k in range(n_neu):
                    s_neu += NETV[x][y][t-1] * NETCON[x][y][i][k] * beta(x, y)

    I_in[x][y] = s_inp + s_neu  # total input current to neuron xy
    NETV[x][y][t] = NETV[x][y][t-1] + \
        (dt / tau) * (-NETV[x][y][t-1] + (I_in[x][y] * Rm))

    ax.cla()
    ax.plot(NETV[:, y, NETV[x][y][t], label='activation of neuron xy')
    plt.show()
# ax.bar3d(NETV[0], NETV[1], NETV[2], np.ones(len(NETV[0])), np.ones(
#    len(NETV[1])), np.ones(len(NETV[2])),  color='b', zsort='average')
