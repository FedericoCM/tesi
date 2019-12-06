"""
Plot of beta function
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


# FUNCTION #
def beta(n1, n0, m_d, h, s):
    """ This function is used to generate weights that change with distance
    according to gaussian distrubution. x0, y0 are 2D coordinates of reference
    neuron; x1, y1 represent neuron to calculate distance """

    max_distance = m_d     # side lenght, max distance between neurons
    A = h                  # high(amplitude) of function
    sigma = s              # standard deviation

    # distance between two points
    x_distance = abs(n1[0] - n0[0])
    y_distance = abs(n1[1] - n0[1])

    """ Makes net a closed surface, if distance is too long, count restarts
    from the opposite side """
    if x_distance > (max_distance / 2):
        x_distance = max_distance - x_distance
    if y_distance > (max_distance / 2):
        y_distance = max_distance - y_distance

    # square distance between neurons
    d2 = (x_distance * x_distance) + (y_distance * y_distance)

    # gaussian function
    return np.exp(-d2 * sigma) * A


# EXAMPLES #
h = 1.                               # gaussian high
s = 1.                              # standard deviation

n_neurons = 10                                      # side lenght
x, y = (np.arange(n_neurons), np.arange(n_neurons))  # x,y axes lenght
n0 = (n_neurons / 2, n_neurons / 2)  # reference neuron

# gaussian distribution in function of distance from neuron x0,y0
NETCON = np.zeros([n_neurons, n_neurons])
# for better plot, neuron distance with itself is considered
for i in range(n_neurons):
    for k in range(n_neurons):
        n1 = (i, k)
        if i == k:
            NETCON[i, k] = 0.
        else:
            NETCON[i, k] = beta(n1, n0, n_neurons, h, s)

print(NETCON)
X, Y = np.meshgrid(x, y)        # generate x,y axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# surface with different colours
ax.plot_surface(X, Y, NETCON, rstride=1, cstride=1,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
# grid lines
ax.plot_wireframe(X, Y, NETCON, color='black', linewidth=0.5)
plt.show()
