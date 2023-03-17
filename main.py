import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from math import dist

def new_point(a, b, t):
    return [a[i]*(1-t)+b[i]*t for i in range(0,3)]

def step_size(pts):
    dists = []
    closed_points = [pt for pt in points]
    closed_points.append(closed_points[0])
    for i, pt in enumerate(closed_points[:-1]):
        dists.append(dist(pt, closed_points[i+1]))
    return np.array(dists).min()/10

def new_edge(a,b,ss):
    new_points = []
    sc = int(round(dist(a,b)/ss,0))
    for step in range(0,sc):
        t = step*ss/dist(a,b)
        new_points.append(new_point(a,b,t))
    return new_points

points = [[0,0,2], [3,0,4], [3,4,2], [0,4,4]]

ss = step_size(points)

nps = []
cps = [pt for pt in points]
cps.append(cps[0])
for i, pt in enumerate(cps[:-1]):
    nps += new_edge(pt, cps[i + 1], ss)

x = [pt[0] for pt in nps]
y = [pt[1] for pt in nps]
z = [pt[2] for pt in nps]

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter(x,y,z)
plt.show()

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