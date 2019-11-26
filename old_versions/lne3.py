"""
Esercizio 3
"""

import numpy as np
import matplotlib.pyplot as plt


def current(t1, t2, v, i, j):
    a = int(t1 / sDeltaTime)
    b = int(t2 / sDeltaTime)
    v[a:b, 0] = i
    v[a:b, 1] = j
    return v


def tanh(x):
    if x <= 0:
        a = 0
    else:
        a = np.tanh(x)
    return a


sSimuTimeDura = 60.  # time simulation in s
sDeltaTime = 0.1    # timestep
sSimuStepDura = int(sSimuTimeDura / sDeltaTime)
tt = np.arange(0, sSimuTimeDura, sDeltaTime)
stau1 = 1.
stau2 = 1.
w12 = -1.
w21 = -1.
vActivTime1 = np.zeros(sSimuStepDura)
out1 = np.zeros(sSimuStepDura)
vActivTime2 = np.zeros(sSimuStepDura)
out2 = np.zeros(sSimuStepDura)
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
        (sDeltaTime / stau1) * \
        (-vActivTime1[t - 1] + vIin[t, 0] + (w21 * out2[t - 1]))

    out1[t] = tanh(vActivTime1[t])

    vActivTime2[t] = vActivTime2[t - 1] + \
        (sDeltaTime / stau2) * \
        (-vActivTime2[t - 1] + vIin[t, 1] + (w12 * out1[t - 1]))

    out2[t] = tanh(vActivTime2[t])

plt.subplot(231)
plt.plot(tt, vIin[:, 0])
plt.subplot(232)
plt.plot(tt, vActivTime1)
plt.subplot(233)
plt.plot(tt, out1)
plt.subplot(234)
plt.plot(tt, vIin[:, 1])
plt.subplot(235)
plt.plot(tt, vActivTime2)
plt.subplot(236)
plt.plot(tt, out2)
plt.show()
