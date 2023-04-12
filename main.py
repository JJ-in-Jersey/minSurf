import numpy as np
from numpy.linalg import norm as length
from sympy.geometry import Point
from sympy.geometry.line import Segment

from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plot
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

class VelocitySurface:

    scale = 100
    LINE = 'LINE'
    SURFACE = 'SURFACE'

    @staticmethod
    def new_point(p1: Point, p2: Point, t): return (p1.scale(1-t, 1-t, 1-t)+p2.scale(t, t, t)).evalf()

    @staticmethod
    def step_size(segments): return np.array([s.length for s in segments]).min()/10

    def edge_points(self, p1: Point, p2: Point, ss):
        num_pts = range(1, int(round(p1.distance(p2)/ss, 0)))
        return [self.new_point(p1, p2, i*ss/p1.distance(p2)) for i in num_pts]

    def plot_vector(self, vector, color, style):
        self.ax.plot3D(x_values(vector), y_values(vector), z_values(vector), c=color, linestyle=style)

    def plot_point(self, point, color, mark):
        self.ax.scatter(point[0], point[1], point[2], c=color, marker=mark)

    def show_axes(self):
        xi = np.linspace(self.x_min, self.x_max, 200)
        yi = np.linspace(self.y_min, self.y_max, 200)
        XI, YI = np.meshgrid(xi, yi)
        if self.shape == VelocitySurface.SURFACE:
            ZI = self.surface(XI, YI)

        self.ax.scatter(self.edge_x, self.edge_y, self.edge_z, c='orange', marker='.')
        self.ax.scatter(self.scaled_x, self.scaled_y, self.scaled_z, c='black', marker='.')
        if self.shape == VelocitySurface.LINE:
            self.plot_vector(self.base_vector, 'lightgrey', 'solid')
            self.plot_vector(VelocitySurface.vector(z_intercept(self.base_vector), self.base_vector[0]), 'grey', 'dotted')
        if self.shape == VelocitySurface.SURFACE:
            self.ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10, color='grey', linewidth=0.25)
        plot.show(block=False)
        plot.pause(0.01)

    def projection(self, x, y, vector):
        base_vector = VelocitySurface.vector(z_intercept(vector), vector[1])
        xy_vector = VelocitySurface.vector(z_intercept(vector), [x,y,0])
        magnitude = np.vdot(xy_vector, base_vector)/length(base_vector)
        unit_vector = np.multiply(base_vector, 1/length(base_vector))
        projection = np.multiply(unit_vector, magnitude)
        projection = translate_3D(projection, VelocitySurface.vector(projection[0], base_vector[0]))
        return projection[1]

    def get_velocity(self, point: Point):
        if not isinstance(point, Point): raise TypeError
        if self.shape == VelocitySurface.SURFACE:
            return Point(point[0], point[1], self.surface(point[0], point[1]).tolist()).evalf()
        elif self.shape == VelocitySurface.LINE:
            z = self.projection(tide_point.lat, tide_point.lon, self.base_vector)
            return z

    def __init__(self, *points):
        if len(points) < 2: raise ValueError
        for pt in points:
            if not isinstance(pt, Point): raise TypeError
        self.points = points

        if len(points) == 2:
            self.shape = VelocitySurface.LINE
        else:
            self.shape = VelocitySurface.SURFACE

        scaled_points = [pt.scale(VelocitySurface.scale, VelocitySurface.scale, 1) for pt in points]
        segments = [Segment(pt, scaled_points[i+1]) for i, pt in enumerate(scaled_points[:-1])]

        if self.shape == VelocitySurface.SURFACE:  # close the figure
            segments.append(Segment(scaled_points[-1], scaled_points[0]))
        elif self.shape == VelocitySurface.LINE:
            self.base_vector = Segment[0]

        ss = self.step_size(segments)
        self.ax = plot.axes(projection="3d")

        points = [scaled_points[0]] + [s.points[1] for s in segments]

        edge_points = []
        for i, pt in enumerate(points[:-1]):
            edge_points += [pt] + self.edge_points(pt, points[i+1], ss)

        if self.shape == 'SURFACE':
            edge_points.append(scaled_points[0])
        elif self.shape == 'LINE':
            edge_points.append(scaled_points[-1])

        points = np.array(points).astype(float)
        edge_points = np.array(edge_points).astype(float)

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

p1 = Point(0, 0, 100)
p2 = Point(3, 0, 300)
p3 = Point(3, 4, 200)
p4 = Point(0, 4, 300)

vs = VelocitySurface(p1,p2,p3,p4)
# vs = VelocitySurface(p1,p3)
vs.show_axes()

for wait in range(0,50):
    rand_x = randrange(vs.x_max - vs.x_min) + vs.x_min
    rand_y = randrange(vs.y_max - vs.y_min) + vs.y_min

    if vs.shape == vs.SURFACE:
        vs.plot_point(Point(rand_x, rand_y, 0), 'grey', '.')
        vs.plot_point(vs.get_velocity(Point(rand_x, rand_y, 0)), 'red', '.')
    elif vs.shape == vs.LINE:
        vs.plot_vector(vs.vector(z_intercept(vs.base_vector), np.array([rand_x, rand_y, 0])), 'lightgrey', 'dotted')
        vs.plot_point(np.array([rand_x, rand_y, 0]), 'lightgrey', '.')
        vs.plot_point(pt, 'red', '.')

    plot.show(block=False)
    plot.pause(0.01)
    sleep(2)

plot.show()
