import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import uniform
from soinn import Soinn
from esoinn import ESoinn
from random import choice

# generate data
n = 12000
sigma = 0.6
c = 10.0 * np.random.rand(n) - 5.0
x0 = -7.0
y0 = 0.0
x1 = -10.5
y1 = 0.0
Y = [[normal(x0, sigma), normal(y0, sigma)] for i in range(int(len(c)/5))]
X = [[c[i], np.sin(c[i])+uniform(-0.2, 0.2)] for i in range(len(c))]
Z = [[normal(x1, sigma), normal(y1, sigma)] for i in range(int(len(c)/5))]
Y.extend(X)
Y.extend(Z)
X = np.array(Y)


# initialize SOINN or ESoinn
# s = Soinn()
s = ESoinn()
s.fit(X)

nodes = s.nodes

print(len(nodes))

print("end")


# show SOINN's state
plt.plot(X[:, 0], X[:, 1], 'cx')

for k in s.adjacent_mat.keys():
    plt.plot(nodes[k, 0], nodes[k, 1], 'k', c='blue')
# plt.plot(nodes[:, 0], nodes[:, 1], 'ro')

color = ['black', 'red', 'saddlebrown', 'skyblue', 'magenta', 'green', 'gold']

# label density
# for i in range(len(s.nodes)):
#     plt.text(s.nodes[i][0], s.nodes[i][1], str(s.density[i]), family='serif', style='italic', ha='right', wrap=True)

color_dict = {}

print(len(s.nodes))
print(len(s.node_labels))

for i in range(len(s.nodes)):
    if not s.node_labels[i] in color_dict:
        color_dict[s.node_labels[i]] = choice(color)
    plt.plot(s.nodes[i][0], s.nodes[i][1], 'ro', c=color_dict[s.node_labels[i]])

plt.grid(True)
plt.show()
