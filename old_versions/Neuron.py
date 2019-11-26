"""
This class simulates neuron
"""


# ------------ IMPORT ------------ #
import numpy as np
import matplotlib.pyplot as plt


# ------------ NEURON ------------ #
class Neuron:
    """ This class take voltage in input and compute neuron response
    if membrane potential at time t exceeds treshold neuron fires """

    def __init__(self, **attr):
        # Data structures

        # Mutable attributes
        # lenght of simulation
        if 'duration' in attr:
            self.duration = attr.get('duration')
        else:
            self.duration = 100
        # timestep
        if 'dt' in attr:
            self.dt = attr.get('dt')
        else:
            self.dt = 1.

        # Random attributes
        if attr.get('rand'):
            l_rest = [-75., -60.]               # interval of resting potential
            l_threshold = [-55., -30.]          # interval of threshold
            l_tau = [2., 10.]                   # interval of tau

            self.v_rest = (l_rest[1] - l_rest[0]) * \
                np.random.rand() + l_rest[0]
            self.threshold = (l_threshold[1] - l_threshold[0]
                              ) * np.random.rand() + l_threshold[0]
            self.tau = (l_tau[1] - l_tau[0]) * np.random.rand() + l_tau[0]

        else:
            # resting potential in mV
            if 'v_rest' in attr:
                self.v_rest = attr.get('v_rest')
            else:
                self.v_rest = -70.
            # threshold for spike in mv
            if 'threshold' in attr:
                self.threshold = attr.get('threshold')
            else:
                self.threshold = -40.
            # time costant in ms
            if 'tau' in attr:
                self.tau = attr.get('tau')
            else:
                self.tau = 10.

        # immutable attributes
        self.abs_ref = 5                    # absolute refractory time (ms)
        self.v_spike = 40.                  # voltage during spike
        self.v_reset = -85                  # iperpolarisation after spike (mV)
        self.t_ref = 0                      # refractory counter (changev)

        # Neuron public attribute
        # voltage variation in postinaptic neurons
        self.v_out = 1.

        # neuron voltage during time (mV)
        self.voltage = np.zeros(self.duration)
        # cell starts at resting potential
        self.voltage[0] = self.v_rest

        # history of spikes
        self.fire = np.zeros(self.duration, dtype=int)
        # input current to neuron at time t (sum of other neurons activation)
        self.v_in = np.full(self.duration, self.v_rest)

    def changev(self, t):
        """ Modify membrane potential, if voltage > treshold
        we have spike and ripolarisation """

        if t > self.t_ref:
            # compute voltage by bacward euler derivative
            self.voltage[t] = self.voltage[t - 1] + \
                (self.dt / self.tau) * \
                (-self.voltage[t - 1] + self.v_in[t])

            # do spike and ripolarisation
            if self.voltage[t] >= self.threshold:
                self.voltage[t] = self.v_spike
                self.t_ref = t + self.abs_ref
                self.voltage[t + self.abs_ref: t: -1] = self.v_reset
                self.fire[t] = 1

            # set reset potential as min voltage value
            if self.voltage[t] < self.v_reset:
                self.voltage[t] = self.v_reset

    def printatt(self):
        return (self.duration, self.dt, self.v_rest, self.threshold, self.tau)

    def set_v_in(self, vol, pos):
        self.v_in[pos] = vol


# ------------ MAIN ------------ #
if __name__ == "__main__":
    a = Neuron(duration=200, rand=True)
    dur = np.arange(1, 200)
    for t in range(100, 150):
        a.set_v_in(-30, t)

    for t in dur:
        a.changev(t)

    print(a.printatt())
    b = a.voltage
    c = a.v_in
    plt.plot(b)
    plt.plot(c)
    plt.show()
