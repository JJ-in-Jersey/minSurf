import numpy as np
from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plt
from math import dist

def new_point(a, b, t):
    return [a[i]*(1-t)+b[i]*t for i in range(0,3)]

def step_size(pts):
    pts_xy = np.multiply(np.append(pts, [pts[0]], axis=0), [1,1,0])  # close the figure, zero out z values
    return np.array([dist(pt, pts_xy[i+1]) for i, pt in enumerate(pts_xy[:-1])]).min()/10

def new_edge(a,b,ss):
    new_points = []
    for step in range(0, int(round(dist(a,b)/ss,0))):
        new_points.append(new_point(a,b,step*ss/dist(a,b)))
    return new_points

# points = [[0,0,2], [3,0,4], [3,4,2], [0,4,4]]
input_points = np.array([[0,0,1], [3,0,3], [3,4,2], [0,4,3]])
input_points = np.array([[0,0,10], [3,0,30], [3,4,20], [0,4,30]])
ss = step_size(input_points)

# z_min = input_points[:,2].min()
# lowered_point = np.add(input_points,[0,0,-1*z_min])

scale = 100
points = np.multiply(input_points,[scale, scale, 1])
ss = ss*scale

nps = []
clpts = np.append(points, [points[0]], axis=0)
for i, pt in enumerate(clpts[:-1]):
    nps += new_edge(pt, clpts[i + 1], ss)

nps = np.array(nps)

x = nps[:,0]
y = nps[:,1]
z = nps[:,2]

rbf = RBF(x,y,z, function='thin_plate', smooth=100.0, mode='1-D')

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter(x, y, z, c='red')
# plt.show()

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