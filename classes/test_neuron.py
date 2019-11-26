import neuron as nrn
import matplotlib.pyplot as plt

a = nrn.Neuron(duration=200, rand=True, name=(0, 0))
z = nrn.Neuron(duration=200, rand=True, name=(0, 1))
grid = [a, z]
report = []
for t in range(100, 150):
    a.set_v_in(t, 50)
    z.set_v_in(t, 50)

for t in range(1, 200):
    report = list(
        filter(None, map(lambda x: x.change_v(t, spike=False), grid)))

    print(report)

print(a, z)
print(a.fire, z.fire)
b = [a.voltage, z.voltage]
c = [a.v_in, z.v_in]

# for i, k in b, c:
#     plt.plot(i)
#     plt.plot(k)
#     plt.show()
plt.plot(b[0])
plt.plot(b[1])
plt.show()
plt.plot(c[0])
plt.plot(c[1])
plt.show()
