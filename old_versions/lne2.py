"""
Esercizio 2
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb


def current(t1, t2, v, i, j):
    a = int(t1 / sDeltaTime)
    b = int(t2 / sDeltaTime)
    v[a:b, 0] = i
    v[a:b, 1] = j
    return v


sSimuTimeDura = 60.  # time simulation in s
sDeltaTime = 0.01    # timestep
sSimuStepDura = int(sSimuTimeDura / sDeltaTime)
stau = 1.
stemp = 0.1
slearn = 1.
vActivTime1 = np.zeros(sSimuStepDura)
vActivTime2 = np.zeros(sSimuStepDura)
sNeu = 2
vIin = np.zeros([sSimuStepDura, sNeu])
current(10., 15., vIin, 0., 1.)
current(15., 20., vIin, 0., 0.)
current(20., 25., vIin, 0.1, 0.9)
current(25., 30., vIin, 0., 0.)
current(30., 35., vIin, 0.2, 0.8)
current(35., 40., vIin, 0., 0.)
current(40., 45., vIin, 0.3, 0.7)
current(45., 50., vIin, 0., 0.)
current(50., 55., vIin, 0.4, 0.6)
current(55., 60., vIin, 0., 0.)


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
