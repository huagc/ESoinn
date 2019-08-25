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
x1 = -7.20
y1 = 0.0
Y = [[normal(x0, sigma), normal(y0, sigma)] for i in range(int(len(c)/4))]
X = [[c[i]/10 - 6.25, np.sin(c[i])/20+uniform(-0.015, 0.015)] for i in range(len(c))]
Z = [[normal(x1, sigma), normal(y1, sigma)] for i in range(int(len(c)/4))]
Y.extend(X)
Y.extend(Z)
X = np.array(Y)


# initialize SOINN or ESoinn
# s = Soinn()
s = ESoinn(iteration_threshold=300)
s.fit(X)

nodes = s.nodes

print(len(nodes))

print("end")