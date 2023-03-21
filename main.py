import numpy as np
from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plt
from math import dist

def new_point(a, b, t):
    return [a[i]*(1-t)+b[i]*t for i in range(0,3)]

def close_loop(pts):
    nps = [pt for pt in pts]
    nps.append(nps[0])
    return nps

def step_size(pts):
    dists = []
    clps = close_loop(pts)
    for i, pt in enumerate(clps[:-1]): dists.append(dist(pt, clps[i+1]))
    return np.array(dists).min()/10

def new_edge(a,b,ss):
    new_points = []
    for step in range(0, int(round(dist(a,b)/ss,0))): new_points.append(new_point(a,b,step*ss/dist(a,b)))
    return new_points

points = [[0,0,2], [3,0,4], [3,4,2], [0,4,4]]
points = [[0,0,0], [3,0,2], [3,4,0], [0,4,2]]
points = [[0,0,0], [3000,0,2], [3000,4000,0], [0,4000,2]]

ss = step_size(points)

nps = []
clpts = close_loop(points)
for i, pt in enumerate(clpts[:-1]):
    nps += new_edge(pt, clpts[i + 1], ss)

x = np.array([pt[0] for pt in nps])
y = np.array([pt[1] for pt in nps])
z = np.array([pt[2] for pt in nps])

rbf = RBF(x,y,z, function='thin_plate', smooth=100.0, mode='1-D')

fig = plt.figure()
ax = plt.axes(projection="3d")

x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()
xi = np.linspace(x_min, x_max, 200)
yi = np.linspace(y_min, y_max, 200)
XI, YI = np.meshgrid(xi, yi)
ZI = rbf(XI,YI)

ax.scatter(x, y, z, c='red')
ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10)

plt.show()