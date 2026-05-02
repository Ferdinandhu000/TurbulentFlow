import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nx, ny = 30, 20
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
Z = np.sin(X/5) + np.cos(Y/5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contourf(X, Y, Z, zdir='z', offset=0, levels=10)
plt.savefig('test_contourf.png')
print("Test passed!")
