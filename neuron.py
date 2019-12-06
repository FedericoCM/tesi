"""
This class simulates neuron
"""


# ------------ IMPORT ------------ #
import numpy as np


# ------------ NEURON ------------ #
class Neuron:
    """ This class takes voltage in input and computes neuron response
    if membrane potential at time t exceeds treshold neuron fires.
    Is very important set a global lenght of simulation
    and pass time to functions """

    # DEFAULT PARAMETERS #
    """ Here default values are setted. Usually these parameters
    are modified during instance creation because they change according
    to simulation """
    # simulation parameters
    dt = 1.                     # timestep(ms)
    duration = 100              # lenght of simulation(ms)

    # neuron specific attributes
    """ In a net name represents neuron position in linear array in form:
    (layer, index). Coordinates represent neuron position in 2D hypothetical
    plane in form: (layer, x, y) """
    name = None                 # neuron name
    coordinates = None          # 2D coordinates
    # types = None              # could be used for different types of neuron
    v_rest = -70.               # resting potential(mV)
    threshold = -40.            # threshold for spike(mV)
    tau = 10.                   # time costant
    rand = False                # used for random attributes
    """ The following two attributes is used to define
    excitatory or inibitory neurons. If the first is false
    neuron is excitatory. The second determines the amplitude
    of postinaptic potential; usually this function is carried out
    by weights, so you can leave + -1. Change this option could be useful
    to simulate large neurons with more prominence in PSP """
    gaba = False                # is this neuron gabaergic or glutamatergic?
    v_out = 1.                  # PSP amplitude(mV)

    # GENERIC ATTRIBUTES #
    """ These values are valid for a wide number of neurons
    or aren't important for simulation. Usually you can leave
    them as they are """
    abs_ref = 5                 # absolute refractory time(ms)
    v_spike = 40.               # spike peak(mV)
    v_reset = -85.              # iperpolarisation after spike(mV)

    def __init__(self, **attr):
        """ This class takes the following parameters:
        (duration, dt, v_rest, threshold, tau, gaba, rand) """
        # PARAMETERS #
        self.__dict__.update(attr)  # update attributes values

        # if rand=True generates random parameters
        if attr.get('rand'):
            l_rest = [-75., -60.]       # interval of resting potential
            l_threshold = [-55., -30.]  # interval of threshold
            l_tau = [2., 10.]           # interval of tau

            self.v_rest = (l_rest[1] - l_rest[0]) * \
                np.random.rand() + l_rest[0]
            self.threshold = (l_threshold[1] - l_threshold[0]
                              ) * np.random.rand() + l_threshold[0]
            self.tau = (l_tau[1] - l_tau[0]) * np.random.rand() + l_tau[0]

        # if you want randomize v_out decomment this region
            # if attr.get('gaba'):
            #     l_v_out = [-10., -0.1]
            # else:
            #     l_v_out = [0.1, 20.]
            # self.v_out = (l_v_out[1] - l_v_out[0]) * \
            #     np.random.rand() + l_v_out[0]

        # if gaba=True sets negative v_out
        if attr.get('gaba'):
            self.v_out = -1.

        # DATA STRUCTURES #
        """ Here are stored data during simulation """
        self.t_ref = 0          # refractory counter(see change_v)

        # voltage during time (mV)
        self.voltage = np.zeros(self.duration)
        self.voltage[0] = self.v_rest  # cell starts at resting potential

        # input current
        """ v_in determines the limit to wich tend v during time.
        It starts full of resting potential value.
        Summation of input currents changes internal state of neuron """
        self.v_in = np.full(self.duration, self.v_rest)

        # ---Following attributes are used only in spike mode--- #
        # dictionary of neurons reached by this neuron and associated weights
        self.connections = {}

        # history of spikes
        self.fire = np.zeros(self.duration, dtype=int)

    # CONNECTIONS FUNCTION #
    def add_connection(self, dictionary):
        """ Add dictionary voice. Key is the name of output neuron
        and value is weight linked with that neuron """
        self.connections.update(dictionary)

    # VOLTAGE VARIATION FUNCTIONS #
    def set_v_in(self, t, voltage):
        """ Used to insert current in v_in at time t """
        self.v_in[t] += voltage

    def change_v(self, t, spike=True):
        """ Modify membrane potential, if voltage > treshold
        we have spike and ripolarisation. This function required timestep
        and should be used in for loops. If spike flag is False Integrated
        and fire became standar leaky simulation """
        # if t<t_ref neuron is in its refractory period and it can't fires
        if t > self.t_ref:
            # computes voltage by bacward euler derivative
            self.voltage[t] = self.voltage[t - 1] + \
                (self.dt / self.tau) * \
                (-self.voltage[t - 1] + self.v_in[t])

            # sets reset potential as min voltage value
            if self.voltage[t] < self.v_reset:
                self.voltage[t] = self.v_reset

            if spike:
                # spike and ripolarisation
                if self.voltage[t] >= self.threshold:
                    self.voltage[t] = self.v_spike
                    self.t_ref = t + self.abs_ref
                    self.voltage[t + self.abs_ref: t: -1] = self.v_reset
                    self.fire[t] = 1
                    # returns neuron's index for firing register
                    return self.name

    # PRINT AND PLOT FUNCTIONS #
    def __str__(self):
        """ Return some object attributes when print is used """
        attributes = {'name': self.name, 'coordinates': self.coordinates,
                      'v_rest': self.v_rest, 'threshold': self.threshold,
                      'tau': self.tau, 'gaba': self.gaba, 'v_out': self.v_out}
        attributes = str(attributes)
        return attributes

    def raster(self):
        """ Saves timesteps when neuron fires. Used for raster plot """
        r = [t for t, f in enumerate(self.fire) if f == 1]
        return r

    def mfreq(self, t, period):
        """ Returns fire medium frequency into given interval """
        s = sum(self.fire[t - period: t])
        s /= period
        return s
