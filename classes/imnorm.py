from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = Image.open('a.jpg')
im = im.convert('L')
data = np.array(im, dtype="float32")
data /= 255

print(data)
im = Image.fromarray(data, "L")
im.save("p.jpg")
