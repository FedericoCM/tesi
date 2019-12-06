"""
This class transforms images in input for net
"""


# ------------ IMPORT ------------ #
from PIL import Image
import numpy as np


# ------------ RETINA ------------ #
class Retina:
    def __init__(self, dimensions):
        # PARAMETERS #
        self.layer_dimensions = dimensions  # width, high of input layer
        self.visual_inp = self.norm_image()

    def vision(self):
        """ Returns array with same dimensions of input layer.
        Values are normalised between 0-1. In spike mode input is
        converted in frequency """
        return self.visual_inp

    # def convert2period(self, inp):
    #     """ Converts input array in fire period for spiking neurons """
    #     period = 1 / (inp * self.max_freq)
    #     return period

    def norm_image(self, img='a.jpg'):
        """ Converts image in array with normalised values """
        im = Image.open(img)
        im = im.convert('L')
        im = im.resize(self.layer_dimensions)
        data = np.array(im, dtype=float)
        data /= 255
        return data
