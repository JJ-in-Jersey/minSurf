import numpy as np
from numpy.linalg import norm as length
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

    def scale(self, x, y, z):
        return TidePoint(self.lat*x, self.lon*y, self.velo*z)

class TidePoints:
    def __init__(self, *tide_points):
        self.size = 0
        self.index = 0
        self.points = None
        if tide_points:
            for pt in tide_points:
                if not isinstance(pt, TidePoint): raise TypeError
            self.points = np.asarray(tide_points)
            self.size = len(tide_points)

    def add(self, tide_point: TidePoint):
        if not isinstance(tide_point, TidePoint): raise TypeError
        if self.points is not None:
            self.points = np.append(self.points, np.asarray(tide_point))
        else:
            self.points = np.asarray(tide_point)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.size:
            tp = self.points[self.index]
            self.index += 1
            return tp
        raise StopIteration

class VelocitySurface:

    scale = 100

    @staticmethod
    def vector(start: TidePoint, end: TidePoint):
        if not isinstance(start, TidePoint) or not isinstance(end, TidePoint): raise TypeError
        return np.array([start.point, end.point])

    @staticmethod
    def new_point(a, b, t):
        return [a[i]*(1-t)+b[i]*t for i in range(0,3)]

    @staticmethod
    def step_size(vectors): return np.array([length(v) for v in vectors]).min()/10

    def edge_points(self, a, b, ss):
        edge_points = []
        number_of_points = int(round(dist(a,b)/ss,0))
        for step in range(1, number_of_points):
            edge_points.append(self.new_point(a,b,step*ss/dist(a,b)))
        return edge_points

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
        scaled_tide_points = TidePoints()
        for pt in tide_points:
            scaled_tide_points.add(pt.scale(VelocitySurface.scale, VelocitySurface.scale, 1))
        vectors = [VelocitySurface.vector(pt, scaled_tide_points.points[i+1]) for i, pt in enumerate(scaled_tide_points.points[:-1])]
        if len(vectors) > 1: vectors.append(np.array([vectors[-1][1], vectors[0][0]]))  # close the figure
        ss = self.step_size(vectors)
        self.ax = plot.axes(projection="3d")

        points = [vectors[0][0]]
        points += [v[1] for v in vectors]
        for i, pt in enumerate(points[:-1]):
            points = [pt.tolist()] + self.edge_points(pt, points[i + 1], ss) + points[i + 1:-1]
        if len(vectors) == 1: points += [vectors[0][1].tolist()]
        new_points = np.array(points)

        self.x = new_points[:, 0]
        self.y = new_points[:, 1]
        self.z = new_points[:, 2]
        self.surface = RBF(self.x, self.y, self.z, function='thin_plate', smooth=100.0, mode='1-D')

p1 = TidePoint(0, 0, 100)
p2 = TidePoint(3, 0, 300)
# p3 = TidePoint(3, 4, 200)
# p4 = TidePoint(0, 4, 300)

# tide_points = TidePoints(p1,p2,p3,p4)
tide_points = TidePoints(p1,p2)
vs = VelocitySurface(tide_points)
vs.show_surface()

for wait in range(0,50):
    rand_x = randrange(vs.x_max - vs.x_min) + vs.x_min
    rand_y = randrange(vs.y_max - vs.y_min) + vs.y_min
    rand_z = vs.surface(rand_x, rand_y)
    vs.show_point(TidePoint(rand_x, rand_y, rand_z))
    sleep(0.05)

plot.show()
