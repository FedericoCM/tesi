"""
Esercizio 1
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb


def current(t1, t2, v, i):
    a = int(t1 / sDeltaTime)
    b = int(t2 / sDeltaTime)
    v[a:b] = i
    return v


sSimuTimeDura = 25.  # time simulation in s
sDeltaTime = 0.01    # timestep
sSimuStepDura = int(sSimuTimeDura / sDeltaTime)
stau = 1.
stemp = 0.1
slearn = 1.
vActivTime1 = np.zeros(sSimuStepDura)
vActivTime2 = np.zeros(sSimuStepDura)
vIin = np.zeros(sSimuStepDura)
current(5., 10., vIin, 1.)
current(10., 15., vIin, 2.)
current(15., 20., vIin, 1.)

for t in range(1, sSimuStepDura):
    vActivTime1[t] = vActivTime1[t - 1] + \
        (sDeltaTime / stau) * (-vActivTime1[t - 1] + vIin[t])
    vActivTime2[t] = vActivTime2[t - 1] + \
        (sDeltaTime / stau) * \
        (-vActivTime2[t - 1] + vIin[t] - vActivTime1[t - 1])

plt.plot(vActivTime1)
plt.plot(vActivTime2)
plt.plot(vIin)
plt.show()
