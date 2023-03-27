import numpy as np
from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plt
from math import dist

def new_point(start, end, dist_from_start):
    scale = dist_from_start/dist(start,end)
    return [start[i]*(1-scale)+end[i]*scale for i in range(0,3)]

def step_size(pts):
    dists = []
    points = pts+[pts[0]]
    for i, pt in enumerate(points[:-1]):
        dists.append(dist(pt, points[i+1]))
    return np.array(dists).min()/10

def new_edge(a,b,ss):
    new_points = []
    for step in range(0, int(round(dist(a,b)/ss,0))): new_points.append(new_point(a,b,step*ss))
    return new_points

points = [[0,0,2], [3,0,4], [3,4,2], [0,4,4]]
points = [[0,0,0], [3,0,2], [3,4,0], [0,4,2]]
points = [[0,0,0],[2000,0,5], [4000,0,0], [4000,1500,5],[4000,3000,0], [2000,3000,15], [0,3000,0],[0,1500,5]]

ss = step_size(points)

nps = []
clpts = points + [points[0]]
for i, pt in enumerate(clpts[:-1]):
    nps += new_edge(pt, clpts[i + 1], ss)

x = np.array([pt[0] for pt in nps])
y = np.array([pt[1] for pt in nps])
z = np.array([pt[2] for pt in nps])

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x, y, z, c='red')
plt.show()

rbf = RBF(x,y,z, function='thin_plate', mode='1-D')

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