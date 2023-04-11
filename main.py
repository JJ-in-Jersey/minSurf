import numpy as np
from numpy.linalg import norm as length
from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plot
from math import dist
from random import randrange
from time import sleep

def z_intercept(vector):
    x0 = vector[0][0]
    y0 = vector[0][1]
    z0 = vector[0][2]
    delta_x = vector[1][0]-x0
    delta_y = vector[1][1]-y0
    delta_z = vector[1][2]-z0
    y = -1*z0*delta_y/delta_z+y0
    x = -1*z0*delta_x/delta_z+x0
    return np.array([x, y, 0])

def translate_3D(points, vector):
    delta_x = vector[1][0] - vector[0][0]
    delta_y = vector[1][1] - vector[0][1]
    delta_z = vector[1][2] - vector[0][2]
    new_points = []
    for pt in points:
        new_points.append([pt[0]+delta_x, pt[1]+delta_y, pt[2]+delta_z])
    return np.array(new_points)

def x_values(points): return points[:, 0]
def y_values(points): return points[:, 1]
def z_values(points): return points[:, 2]

class TidePoint:
    def __init__(self, lat: float, lon: float, velo: float):
        self.point = np.array([lat, lon, velo])
        self.lat = lat
        self.lon = lon
        self.velo = velo

    def scale(self, x, y, z):
        return TidePoint(self.lat*x, self.lon*y, self.velo*z)

class TidePoints:
    def update(self):
        for pt in self.tide_points:
            if not isinstance(pt, TidePoint): raise TypeError
        self.size = len(self.tide_points)
        # self.tide_points = np.asarray(tide_points)
        self.points = np.array([tp.point for tp in self.tide_points])
        self.first_tide_point = self.tide_points[0]
        self.last_tide_point = self.tide_points[-1]
        self.first_point = self.points[0]
        self.last_point = self.points[-1]

    def __init__(self, *tide_points):
        self.index = 0
        self.size = 0
        self.tide_points = self.points = None
        self.first_tide_point = self.last_tide_point = self.first_point = self.last_point = None
        if tide_points:
            self.tide_points = tide_points
            self.update()

    def add(self, tide_point: TidePoint):
        if not isinstance(tide_point, TidePoint): raise TypeError
        if self.tide_points is not None:
            self.tide_points = np.append(self.tide_points, tide_point)
        else:
            self.tide_points = np.asarray([tide_point])
        self.update()

    def __iter__(self): return self
    def __next__(self):
        if self.index < self.size:
            tp = self.tide_points[self.index]
            self.index += 1
            return tp
        raise StopIteration

class VelocitySurface:

    scale = 100
    LINE = 'LINE'
    SURFACE = 'SURFACE'

    @staticmethod
    def vector(start, finish):
        begin = end = None
        if isinstance(start, TidePoint):
            begin = start.point
        elif isinstance(start, np.ndarray) and len(start) == 3:
            begin = start
        elif isinstance(start, list) and len(start) ==3:
            begin = np.array(start)

        if isinstance(finish, TidePoint):
            end = finish.point
        elif isinstance(finish, np.ndarray) and len(finish) == 3:
            end = finish
        elif isinstance(finish, list) and len(finish) == 3:
            end = np.array(finish)

        return np.array([begin, end])

    @staticmethod
    def new_point(a, b, t): return np.array([a[i]*(1-t)+b[i]*t for i in range(0,3)])

    @staticmethod
    def step_size(vectors): return np.array([length(v) for v in vectors]).min()/10

    def edge_points(self, a, b, ss):
        edge_points = []
        number_of_points = int(round(dist(a,b)/ss,0))
        for step in range(1, number_of_points):
            edge_points.append(self.new_point(a,b,step*ss/dist(a,b)))
        return edge_points

    def plot_vector(self, vector, color):
        self.ax.plot3D(x_values(vector), y_values(vector), z_values(vector), c=color, linestyle='dotted')

    def plot_point(self, point, color, mark):
        self.ax.scatter(point[0], point[1], point[2], c=color, linestyle='dotted', marker=mark)

    def show_axes(self):
        xi = np.linspace(self.x_min, self.x_max, 200)
        yi = np.linspace(self.y_min, self.y_max, 200)
        XI, YI = np.meshgrid(xi, yi)
        if self.shape == VelocitySurface.SURFACE:
            ZI = self.surface(XI, YI)

        self.ax.scatter(self.edge_x, self.edge_y, self.edge_z, c='orange', marker='.')
        self.ax.scatter(self.scaled_x, self.scaled_y, self.scaled_z, c='black', marker='.')
        if self.shape == VelocitySurface.LINE:
            self.ax.plot3D(self.scaled_x, self.scaled_y, self.scaled_z, color='grey', linewidth=0.25)
        if self.shape == VelocitySurface.SURFACE:
            self.ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10, color='grey', linewidth=0.25)
        plot.show(block=False)
        plot.pause(0.01)

    def show_point(self, tide_point):
        if not isinstance(tide_point, TidePoint): raise TypeError
        z = tide_point.velo
        if self.shape == VelocitySurface.SURFACE:
            self.plot_point(tide_point.point, 'red', 'o')
            self.plot_point(tide_point.scale(1,1,0).point, 'lightgrey', 'o')
        elif self.shape == VelocitySurface.LINE:
            self.plot_point(tide_point.point, 'red', '0')

        plot.show(block=False)
        plot.pause(0.01)

    def projection(self, x, y, vector):
        z_point = z_intercept(vector)
        self.plot_vector(VelocitySurface.vector(z_point, vector[0]), 'lightgrey')
        vector = VelocitySurface.vector(z_point, vector[1])
        xy_vector = VelocitySurface.vector(z_point, [x,y,0])
        self.plot_vector(xy_vector, 'lightgrey')
        magnitude = np.vdot(xy_vector, vector)/length(vector)
        unit_vector = np.multiply(vector, 1/length(vector))
        projection = np.multiply(unit_vector, magnitude)
        projection = translate_3D(projection, VelocitySurface.vector(projection[0], vector[0]))
        return projection[1]

    def get_velocity(self, tide_point):
        if not isinstance(tide_point, TidePoint): raise TypeError
        rtp = None
        if self.shape == VelocitySurface.SURFACE:
            z = self.surface(tide_point.lat, tide_point.lon)
            return np.array([tide_point.lat, tide_point.lon, z])
        elif self.shape == VelocitySurface.LINE:
            z = self.projection(tide_point.lat, tide_point.lon, self.vector)
            return z

    def __init__(self, tide_points):
        if tide_points.size < 2: raise ValueError
        if not isinstance(tide_points, TidePoints): raise TypeError
        self.edge_x = self.edge_y = self.edge_z = 0
        self.scaled_x = self.scaled_y = self.scaled_z = 0
        self.x_min = self.x_max = self.y_min = self.y_max = 0
        self.shape = self.vector = self.surface = None
        self.tide_points = tide_points

        if tide_points.size == 2:
            self.shape = VelocitySurface.LINE
        else:
            self.shape = VelocitySurface.SURFACE

        scaled_tide_points = TidePoints()
        for pt in tide_points:
            scaled_tide_points.add(pt.scale(VelocitySurface.scale, VelocitySurface.scale, 1))

        vectors = [VelocitySurface.vector(pt, scaled_tide_points.points[i+1]) for i, pt in enumerate(scaled_tide_points.points[:-1])]
        if self.shape == VelocitySurface.SURFACE:  # close the figure
            vectors.append(VelocitySurface.vector(scaled_tide_points.last_tide_point, scaled_tide_points.first_tide_point))
        elif self.shape == VelocitySurface.LINE:
            self.vector = vectors[0]

        ss = self.step_size(vectors)
        self.ax = plot.axes(projection="3d")

        points = [scaled_tide_points.first_point]
        points += [v[1] for v in vectors]

        edge_points = []
        for i, pt in enumerate(points[:-1]):
            edge_points += [pt] + self.edge_points(pt, points[i + 1], ss)
        if self.shape == 'SURFACE':
            edge_points.append(scaled_tide_points.first_point)
        elif self.shape == 'LINE':
            edge_points.append(scaled_tide_points.last_point)

        points = np.array(points)
        edge_points = np.array(edge_points)

        self.scaled_x = x_values(points)
        self.scaled_y = y_values(points)
        self.scaled_z = z_values(points)

        self.edge_x = x_values(edge_points)
        self.edge_y = y_values(edge_points)
        self.edge_z = z_values(edge_points)

        self.x_min = int(round(x_values(points).min(), 0))
        self.x_max = int(round(x_values(points).max(), 0))
        self.y_min = int(round(y_values(points).min(), 0))
        self.y_max = int(round(y_values(points).max(), 0))

        if self.shape == VelocitySurface.SURFACE:
            self.surface = RBF(self.edge_x, self.edge_y, self.edge_z, function='thin_plate', smooth=100.0)

p1 = TidePoint(0, 0, 100)
p2 = TidePoint(3, 0, 300)
p3 = TidePoint(3, 4, 200)
p4 = TidePoint(0, 4, 300)

tide_points = TidePoints(p1,p2,p3,p4)
# tide_points = TidePoints(p1,p3)
# tide_points = TidePoints(p1,p3, p4)
vs = VelocitySurface(tide_points)
vs.show_axes()

for wait in range(0,50):
    rand_x = randrange(vs.x_max - vs.x_min) + vs.x_min
    rand_y = randrange(vs.y_max - vs.y_min) + vs.y_min

    pt = vs.get_velocity(TidePoint(rand_x, rand_y, 0))
    vs.show_point(TidePoint(pt[0], pt[1], pt[2]))
    sleep(0.05)

plot.show()
