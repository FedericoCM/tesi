import numpy as np
import matplotlib.pyplot as plt
import spike as sp

n = sp.Neuron()
print(n.fire)
np.resize(n.v_in, (1, 100))
n.v_in[:] = -30.
np.resize(n.fire, (1, 100))
np.resize(n.voltage, (1, 100))
f = 0.
for t in range(100):
    n.changev(t)
    f += n.fire[t]
print(f)
print(n.fire)

plt.plot(n.voltage)
plt.show()
plt.plot(n.fire)
plt.show()
