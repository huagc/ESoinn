import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.random import uniform
from soinn import Soinn
from esoinn import ESoinn
from random import choice

# generate data
n = 10000
sigma = 0.02
c = 10.0 * np.random.rand(n) - 5.0
x0 = -7.05
y0 = 0.0
x1 = -7.155
y1 = 0.0
Y = [[normal(x0, sigma), normal(y0, sigma)] for i in range(int(len(c)/3))]
X = [[c[i]/10 - 6.25, np.sin(c[i])/10+uniform(-0.02, 0.02)] for i in range(len(c))]
Z = [[normal(x1, sigma), normal(y1, sigma)] for i in range(int(len(c)/3))]
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
#     str_tmp = str(s.density[i]) + " " + str(s.N[i])
#     plt.text(s.nodes[i][0], s.nodes[i][1], str_tmp, family='serif', style='italic', ha='right', wrap=True)

color_dict = {}

print(len(s.nodes))
print(len(s.node_labels))

for i in range(len(s.nodes)):
    if not s.node_labels[i] in color_dict:
        color_dict[s.node_labels[i]] = choice(color)
    plt.plot(s.nodes[i][0], s.nodes[i][1], 'ro', c=color_dict[s.node_labels[i]])

plt.grid(True)
plt.show()
