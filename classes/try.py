import neuron
import numpy as np


class bo:
    """ pos dizionario con le posizioni
    fire set con i neuroni che scaricano
    map lista con i collegamenti
    w dizionario del neurone con i pesi delle connessioni con gli altri neuroni
    """

    def __init__(self, **attr):
        self.name = attr.get('name')
        self.coo = attr.get('coo')

    def change_name(self, cose):
        self.name = cose


# dizio = {(0, 1): 3, (2, 2): 2}
# # a = bo(**dizio)
# # print(a.__dict__)

# # for key, value in dizio.items():
# #     print(key[0])
# grid = []
# firing = []
# for i in range(3):
#     n = neuron.Neuron()
#     grid.append(n)

# a = np.random.rand(3, 2)
# print(a)
# a = a[:, 0]
# print(a)
dimensions = [(2, 2), (2, 1), (1, 1)]
a = (0, 1)
for b in a:
    print(dimensions[b])
