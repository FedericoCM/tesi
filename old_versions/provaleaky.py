"""
Prima prova per un modello leaky neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# parameters and variables
T = 50.                           # total time simulation (ms)
dt = 0.2                          # timestep (ms)
time = np.arange(0, T+dt, dt)     # time array
lt = len(time)                    # total timestep
# neurons properties
Vm = np.zeros(lt)                 # potential over time (mV)
Rm = 1.                           # membrane resistance (kOhms)
Cm = 10.                          # membrane capacity (uF)
tau = Rm * Cm                     # time costant (ms)

# input stimulus
I_stim = 15.                      # input current (nA)
I_in = np.zeros(lt)
I_in = np.array([norm.pdf(i/10. - 1.) + norm.pdf(i/10. - 30.)
                 - 0.02/(i+1) for i in range(lt)])  # gaussian stimulus in input


# iteration
for i in range(1, lt):
    Vm[i] = Vm[i-1] + (dt / tau) * (-Vm[i-1] + (I_stim * Rm))
    print(Vm[i])

# plot
plt.plot(time, Vm)
for i in range(lt):
    if I_in[i] > 0.2:
        plt.axvline(i, color="red")
   # elif I_in[i] < 0.2:
    #    plt.axvline(i, color="green")

plt.title("Leaky neurons expample")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.show()
