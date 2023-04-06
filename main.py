import numpy as np
from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plot
from math import dist
from random import randrange
from time import sleep


class TidePoint:
    def __init__(self, lat: float, lon: float, velo: float):
        self.point = np.array([lat, lon, velo])
        self.lat = lat
        self.lon = lon
        self.velo = velo

class TidePoints:
    def __init__(self, *points):
        self.points = np.empty(shape=[len(points),3])
        for i, pt in enumerate(points):
            if not isinstance(pt, TidePoint): raise TypeError
            self.points[i] = pt.point

class VelocitySurface:

    scale = 100

    @staticmethod
    def new_point(a, b, t):
        return [a[i]*(1-t)+b[i]*t for i in range(0,3)]

    @staticmethod
    def step_size(pts):
        pts_xy = np.multiply(np.append(pts, [pts[0]], axis=0), [1,1,0])  # close the figure, zero out z values
        return np.array([dist(pt, pts_xy[i+1]) for i, pt in enumerate(pts_xy[:-1])]).min()/10

    def new_edge(self, a, b, ss):
        new_points = []
        for step in range(0, int(round(dist(a,b)/ss,0))):
            new_points.append(self.new_point(a,b,step*ss/dist(a,b)))
        return new_points

    def show_surface(self):
        self.x_min = int(round(self.x.min(), 0))
        self.x_max = int(round(self.x.max(), 0))
        self.y_min = int(round(self.y.min(), 0))
        self.y_max = int(round(self.y.max(), 0))

        xi = np.linspace(self.x_min, self.x_max, 200)
        yi = np.linspace(self.y_min, self.y_max, 200)
        XI, YI = np.meshgrid(xi, yi)
        ZI = self.surface(XI, YI)

        self.ax.scatter(self.x, self.y, self.z, c='red', marker='.')
        self.ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10, color='gray', linewidth=0.25)
        plot.show(block=False)
        plot.pause(0.01)

    def show_point(self, tide_point):
        if not isinstance(tide_point, TidePoint): raise TypeError
        # ax = plot.axes(projection="3d")
        self.ax.scatter(tide_point.lat, tide_point.lon, tide_point.velo, c='blue')
        plot.show(block=False)
        plot.pause(0.01)

    def __init__(self, tide_points):
        if not isinstance(tide_points, TidePoints): raise TypeError
        self.x_min = self.x_max = self.y_min = self.y_max = 0
        ss = self.step_size(tide_points.points)*VelocitySurface.scale
        points = np.multiply(tide_points.points,[VelocitySurface.scale, VelocitySurface.scale, 1])  # scale up the x, y values
        self.ax = plot.axes(projection="3d")

        new_points = []
        closed_figure = np.append(points, [points[0]], axis=0)  # close figure
        for i, pt in enumerate(closed_figure[:-1]):
            new_points += self.new_edge(pt, closed_figure[i + 1], ss)
        new_points = np.array(new_points)

        self.x = new_points[:, 0]
        self.y = new_points[:, 1]
        self.z = new_points[:, 2]
        self.surface = RBF(self.x, self.y, self.z, function='thin_plate', smooth=100.0, mode='1-D')

p1 = TidePoint(0, 0, 100)
p2 = TidePoint(3, 0, 300)
# p3 = TidePoint(3, 4, 200)
# p4 = TidePoint(0, 4, 300)

# tps = TidePoints(np.array([p1,p2,p3,p4]))
# tps = TidePoints(p1,p2,p3,p4)
tps = TidePoints(p1,p2)
vs = VelocitySurface(tps)
vs.show_surface()

for wait in range(0,20):
    rand_x = randrange(vs.x_max - vs.x_min) + vs.x_min
    rand_y = randrange(vs.y_max - vs.y_min) + vs.y_min
    rand_z = vs.surface(rand_x, rand_y)
    vs.show_point(TidePoint(rand_x, rand_y, rand_z))
    print(wait)
    sleep(0.5)

plot.show()
