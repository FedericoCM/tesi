import plotter as pl
import numpy as np

dimensions = [(2, 2), (2, 1), (1, 1)]
n_neurons = [4, 2, 1]
duration = 2

a = []
c = []
for i in n_neurons:
    b = np.random.rand(i, duration) * 2
    a.append(b)
    if i == 4:
        c.append(b)
    else:
        e = []
        c.append(e)


print(a)
print(c)
p = pl.Plotter(duration, dimensions, n_neurons)
p.raster_plot(a)
p.raster_plot(c)
