import numpy as np
from sympy.geometry import Point, Line, Segment, Plane
from sympy import symbols

from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plot
from random import randrange
from time import sleep

class VelocitySurface:

    scale = 100
    LINE = 'LINE'
    SURFACE = 'SURFACE'
    XY_PLANE = Plane(Point(0, 0, 0), normal_vector=[0, 0, 1])

    @staticmethod
    def __step_size(segments): return np.array([s.length for s in segments]).min()/10

    @staticmethod
    def __edge_points(segment: Segment, ss):
        t = symbols('t')
        num_pts = range(0, int(round(segment.length/ss, 0)))
        return [segment.arbitrary_point(t).evalf(subs={t:i*ss/segment.length}) for i in num_pts]

    def plot_segment(self, segment, color, style, weight):
        points = np.array(segment.points).astype(float)
        self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2], c=color, linestyle=style, linewidth=weight)

    def plot_point(self, point, color, mark):
        self.ax.scatter(point.x, point.y, point.z, c=color, marker=mark)

    def show_plot(self):
        xi = np.linspace(self.x_min, self.x_max, 300)
        yi = np.linspace(self.y_min, self.y_max, 300)
        XI, YI = np.meshgrid(xi, yi)
        if self.shape == VelocitySurface.SURFACE: ZI = self.surface(XI, YI)

        self.ax = plot.axes(projection="3d")
        self.ax.scatter(self.edge_x, self.edge_y, self.edge_z, c='orange', marker='.')
        self.ax.scatter(self.scaled_x, self.scaled_y, self.scaled_z, c='black', marker='.')

        if self.shape == VelocitySurface.LINE:
            z_intercept = VelocitySurface.XY_PLANE.intersection(Line(self.range))[0]
            self.plot_point(z_intercept, 'black', '.')
            self.plot_segment(Segment(z_intercept, self.range.p2), 'grey', '--', 0.5)
            self.plot_segment(Segment(z_intercept, self.input_point), 'grey', '--', 0.5)
        if self.shape == VelocitySurface.SURFACE:
            self.ax.plot_wireframe(XI, YI, ZI, rstride=10, cstride=10, color='grey', linewidth=0.25)

        self.plot_segment(Segment(self.input_point, self.output_point), 'grey', '--', 0.5)
        self.plot_point(self.output_point, 'red', 'o')
        plot.show()

    def get_velocity(self, point: Point):
        if not isinstance(point, Point): raise TypeError
        self.input_point = point
        if self.shape == VelocitySurface.SURFACE:
            self.output_point = Point(point.x, point.y, self.surface(point[0], point[1]).tolist()).evalf()
        elif self.shape == VelocitySurface.LINE:
            # self.output_point = Line(self.range).projection(point)
            self.output_point = Line(self.range).perpendicular_line(point).p2
        return self.output_point

    def __init__(self, *points):
        if len(points) < 2: raise ValueError
        for pt in points:
            if not isinstance(pt, Point): raise TypeError
        self.points = points
        self.ax = self.input_point = self.output_point = self.perp_seg = None

        if len(points) == 2: self.shape = VelocitySurface.LINE
        else: self.shape = VelocitySurface.SURFACE

        scaled_points = [pt.scale(VelocitySurface.scale, VelocitySurface.scale, 1) for pt in points]
        segments = [Segment(pt, scaled_points[i+1]) for i, pt in enumerate(scaled_points[:-1])]

        if self.shape == VelocitySurface.SURFACE: segments.append(Segment(scaled_points[-1], scaled_points[0]))  # close the figure
        elif self.shape == VelocitySurface.LINE:
            self.range = segments[0]

        ss = self.__step_size(segments)
        points = [scaled_points[0]] + [s.points[1] for s in segments]
        edge_points = []
        for segment in segments:
            edge_points += self.__edge_points(segment, ss)

        if self.shape == 'SURFACE': edge_points.append(scaled_points[0])
        elif self.shape == 'LINE': edge_points.append(scaled_points[-1])

        points = np.array(points).astype(float)
        edge_points = np.array(edge_points).astype(float)

        self.scaled_x = points[:, 0]
        self.scaled_y = points[:, 1]
        self.scaled_z = points[:, 2]

        self.edge_x = edge_points[:, 0]
        self.edge_y = edge_points[:, 1]
        self.edge_z = edge_points[:, 2]

        self.x_min = int(round(self.scaled_x.min(), 0))
        self.x_max = int(round(self.scaled_x.max(), 0))
        self.y_min = int(round(self.scaled_y.min(), 0))
        self.y_max = int(round(self.scaled_y.max(), 0))

        if self.shape == VelocitySurface.SURFACE:
            self.surface = RBF(self.edge_x, self.edge_y, self.edge_z, function='thin_plate', smooth=100.0)

p1 = Point(0, 0, 100)
p2 = Point(3, 0, 300)
p3 = Point(3, 4, 200)
p4 = Point(0, 4, 300)

vs = VelocitySurface(p1,p2,p3,p4)
# vs = VelocitySurface(p1,p3, p4)
# vs = VelocitySurface(p1,p3)
rand_pt = Point(randrange(vs.x_max - vs.x_min) + vs.x_min, randrange(vs.y_max - vs.y_min) + vs.y_min, 0)
z = vs.get_velocity(rand_pt)
vs.show_plot()
