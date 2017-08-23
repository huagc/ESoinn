import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import uniform
from soinn import Soinn
from esoinn import ESoinn
import copy

# generate data
n = 10000
sigma = 0.07
c = 10 * np.random.rand(n) - 5
X = np.array([[c[i], np.sin(c[i])+uniform(-0.2, 0.2)] for i in range(len(c))])


# initialize SOINN
s = ESoinn()
s.fit(X)

# a = [1, 1, 1, 1]
# d = {1:10, 2:100, 3:101}
# b = a
# b[1] = 10
# for i in range(4):
#     a[i] = 100
# print(a)
# print(b)
# print(np.mean(b))

nodes = s.nodes

print(len(nodes))

print("end")

# for i in list(s.adjacent_mat.keys()):
#     s.adjacent_mat.pop((i[0], i[1]))

# show SOINN's state
plt.plot(X[:, 0], X[:, 1], 'cx')

for k in s.adjacent_mat.keys():
    plt.plot(nodes[k, 0], nodes[k, 1], 'k')
plt.plot(nodes[:, 0], nodes[:, 1], 'ro')


for i in range(len(s.nodes)):
    plt.text(s.nodes[i][0], s.nodes[i][1], str(s.density[i]), family='serif', style='italic', ha='right', wrap=True)

plt.grid(True)
plt.show()
