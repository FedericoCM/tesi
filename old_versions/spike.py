"""
Prima prova di una rete spiking neuron
"""

# ------------ IMPORT ------------ #
import numpy as np
import matplotlib.pyplot as plt


# ------------ FUNCTIONS ------------ #

# Simulate electrode
def current(t1, t2, v, i):
    a = int(t1 / dt)            # start stimulation time
    b = int(t2 / dt)            # stop stimulation time
    v[a:b] = i                  # insert value of stimulation i into v array
    return v


# ------------ VARIABLES ------------ #

# Simulation parameters
time = 250.                   # total time simulation in ms
dt = 1.                         # timestep in ms
duration = int(time / dt)       # total steps during simulation
neurons = 1                     # neurons number

# Neuron properties
Rm = 1.                         # membrane resistance in kOhms
Cm = 10.                        # membrane capacity in uF
tau = Rm * Cm                   # time costant in ms
abs_ref = int(5. / dt)          # absolute refractory time in ms
v_rest = -70.                   # resting potential in mV
v_reset = -85.                  # iperpolarisation after spike in mV
v_spike = 35.                   # threshold for spike in mv
v_max = 70.                     # max voltage during spike

# ------------ DATA STRUCTURES ------------ #

# Voltage during time
V = np.zeros(duration)        # mebrane voltage of neuron during time (mv)
V[0] = v_rest                   # cell starts at resting potential

# electrode input to neuron
I_in = np.full(duration, v_rest)
current(50., 100., I_in, 30.)
current(100., 150., I_in, 50.)
current(150., 200., I_in, 20.)


# ------------ ITERATION ------------ #

for t in range(1, duration):

    # wait refractory time to reset
    if V[t - 1] >= v_spike and V[t - 1] < v_max:
        V[t:t + abs_ref] = v_max
        V[t + abs_ref] = v_reset

    # use Euler derivative to compute membrane potential
    elif V[t - 1] < v_spike:
        V[t] = V[t - 1] + (dt / tau) * (-V[t - 1] + (I_in[t] * Rm))


# ------------ PLOT ------------ #

plt.plot(V)
plt.plot(I_in)
plt.show()
