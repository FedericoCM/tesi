import retina

f = (1 / 60.) / 0.005
a = retina.Retina((2, 2))
b = a.vision()
print(b)
