"""
Creation date: 21/05/19
Author: Federico Cascino Milani
Title: Spiking neuron test
Description: two neurons, excitatory and inibitory, linked togheter
and connected to electrode
"""

# ------------ IMPORT ------------ #
import numpy as np
import matplotlib.pyplot as plt


# ------------ VARIABLES ------------ #

# Simulation parameters
time = 500.                     # total time simulation in ms
dt = 1.                         # timestep in ms
duration = int(time / dt)       # total steps during simulation
n_neurons = 2                   # neurons number


# ------------ CLASSES ------------ #

# Neuron properties
class Neuron:
    abs_ref = int(5. / dt)          # absolute refractory time in ms
    v_rest = -70.                   # resting potential in mV
    v_reset = -85.                  # iperpolarisation after spike in mV
    v_spike = 40.                   # voltage during spike
    threshold = -35.                # threshold for spike in mv

    def __init__(self, v_out, CON, w):
        self.con = CON                     # connections with other neurons
        self.w = w                         # weights of connections
        self.t_ref = 0                     # refractory counter
        self.v_out = v_out                   # postsinaptic potential
        self.voltage = np.zeros(duration)  # neuron voltage during time (mv)
        self.voltage[0] = Neuron.v_rest    # cell starts at resting potential
        self.tau = 10.   # time costant in ms
        self.fire = np.zeros(duration, dtype=int)  # history of spikes

    # modify membrane potential if voltage > treshold
    # we have spike and ripolarisation
    def changev(self, t, v_in):
        if t > self.t_ref:
            self.voltage[t] = self.voltage[t - 1] + \
                (dt / self.tau) * (-self.voltage[t - 1] + v_in)

            if self.voltage[t] >= Neuron.threshold:
                self.voltage[t] = Neuron.v_spike
                self.t_ref = t + Neuron.abs_ref
                self.voltage[t + Neuron.abs_ref: t: -1] = Neuron.v_reset
                self.fire[t] = 1

    def synapse(self, t, neu):
        #neu = 'n_' + str(self.con)
        if neu.fire[t - 1] == 1:
            v_in = neu.v_out * self.w
        else:
            v_in = Neuron.v_rest
        return v_in


# ------------ FUNCTIONS ------------ #

# Simulate electrode. Insert value of stimulation i into v array
def current(t, d, v, i):
    v[t: t + d] = i


# ------------ DATA STRUCTURES ------------ #

# electrode input to neuron
v_ext = np.full(duration, Neuron.v_rest)
# for i in range(251, duration):
#    v_ext[i] = v_ext[i - 1] + 1.5
#    if v_ext[i - 1] >= 50.:
#        v_ext[i] = Neuron.v_rest
current(20, 100, v_ext, -30.)
current(150, 50, v_ext, 0.)
current(220, 30, v_ext, -60.)
current(300, 100, v_ext, 70.)

CON = np.array([None, 0])


# neurons generation
n_0 = Neuron(50., CON[0], 0.)
n_1 = Neuron(-20., CON[1], 6.)
n_1.tau = 18.


# ------------ ITERATION ------------ #

for t in range(1, duration):
    n_0.changev(t, v_ext[t])
    n_1.changev(t, n_1.synapse(t, n_0))


# ------------ PLOT ------------ #

plt.plot(n_0.voltage)
plt.plot(n_1.voltage)
plt.plot(v_ext)
plt.show()
