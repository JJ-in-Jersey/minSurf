import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection="3d")

# x = np.array([0.0, 1.5, 3.0, 3.0, 3.0, 1.5, 0.0, 0.0])
# y = np.array([0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 2.0])
# z = np.array([2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 2.0])
x = np.array([0.0, 3.0, 3.0, 0.0])
y = np.array([0.0, 0.0, 4.0, 4.0])
z = np.array([2.0, 4.0, 2.0, 4.0])

x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()
xi = np.linspace(x_min, x_max, 200)
yi = np.linspace(y_min, y_max, 200)
XI, YI = np.meshgrid(xi, yi)

rbf = Rbf(x,y,z,function='thin-plate',smooth=0.0)
ZI = rbf(XI,YI)

# ax.plot3D(x, y, z, 'red')
# ax.plot3D(XI, YI, ZI, 'red')
ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10)
# ax.scatter3D(XI, YI, ZI, c=z, cmap='cividis')
plt.show()