"""
This file is used to run simulation
"""

# ------------ IMPORT ------------ #
import net


# ------------ PARAMETERS ------------ #
def create_dict(*args):
    return dict(((k, eval(k)) for k in args))


duration = 5000
dimensions = [(10, 10), (10, 10), (10, 10)]
refresh_rate = 0.06
input_pulse = 100.
spike = True
omicron = 200.

net_attr = create_dict('duration', 'dimensions', 'refresh_rate',
                       'input_pulse', 'spike', 'omicron')

dt = 1.
rand = True
gaba = True

neuronex_attr = create_dict('dt', 'rand')
neuronib_attr = create_dict('dt', 'gaba', 'rand')


# ------------ SIMULATION ------------ #
rete = net.Net(**net_attr)
layers = list(range(len(dimensions)))

# net generation
for layer in layers:
    if layer < max(layers):
        rete.grid_gen(layer, **neuronex_attr)
    else:
        rete.grid_gen(layer, **neuronib_attr)

# net connections
rete.all2all(0, 1)
rete.intraconnection(1)
rete.all2all(1, 2)
rete.intraconnection(2)
rete.all2all(2, 1)

# net activation
rete.activation()

# collect data
rete.collect_raster()
print(rete.M_freq)

# plot
rete.printer.raster_plot(rete.rasplot)
