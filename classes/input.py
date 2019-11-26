"""
This class transforms images in input for net
"""


# ------------ IMPORT ------------ #
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import net


# ------------ INPUT ------------ #
class Input:
    def __init__(self, net, layer_out):
        self.net_dimensions = net.dimensions
        self.layer_out = layer_out
        self.input_array = [self.resize(self.net_dimensions[l])
                            for l in layer_out]

    def convert2freq(self, inp, duration):

    def resize(self, newsize):           # inserire immagine come parametro
        im = Image.open('a.jpg')
        im = im.convert('L')
        im = im.resize(newsize)
        # im.save('o.jpg')
        data = np.array(im, dtype=float)
        data /= 255
        return data


# MAIN
if __name__ == '__main__':
    layer_out = (0, 1)
    a = Input(net.Net(dimensions=[(100, 100), (50, 50), (1, 1)]), layer_out)
    print(a.input_array)
