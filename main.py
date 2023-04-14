import numpy as np
from sympy.geometry import Point, Line, Segment, Plane
from sympy import symbols

from scipy.interpolate import Rbf as RBF
from matplotlib import pyplot as plot
from random import randrange
from time import sleep


# noinspection PyUnresolvedReferences
class VelocitySurface:

    scale = 100
    LINE = 'LINE'
    SURFACE = 'SURFACE'
    XY_PLANE = Plane(Point(0, 0, 0), normal_vector=[0, 0, 1])

    @staticmethod
    def __step_size(segments): return np.array([s.length for s in segments]).min()/10

    def __edge_points(self, segments, ss, first_point, last_point):
        t = symbols('t')
        edge_points = []
        for segment in segments:
            num_pts = range(0, int(round(segment.length / ss, 0)))
            edge_points += [segment.arbitrary_point(t).evalf(subs={t: pt*ss/segment.length}) for pt in num_pts]
        if self.shape == VelocitySurface.SURFACE:
            np.append(edge_points, first_point)
        elif self.shape == VelocitySurface.LINE:
            np.append(edge_points, last_point)
        return np.array(edge_points).astype(float)

    def __plot_segment(self, segment: Segment, color: str, style: str, weight: float):
        points = np.array(segment.points).astype(float)
        self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2], c=color, linestyle=style, linewidth=weight)

    def __plot_point(self, point: Point, color: str, mark: str):
        self.ax.scatter(point.x, point.y, point.z, c=color, marker=mark)

    def show_plot(self):
        xi = np.linspace(min(self.x_limits), max(self.x_limits), 300)
        yi = np.linspace(min(self.y_limits), max(self.y_limits), 300)
        XI, YI = np.meshgrid(xi, yi)

        # self.ax = plot.axes(projection="3d")
        self.ax.scatter(self.scaled_edge_points[0], self.scaled_edge_points[1], self.scaled_edge_points[2], c='orange', marker='.')
        self.ax.scatter(self.scaled_input[0], self.scaled_input[1], self.scaled_input[2], c='black', marker='.')

        if self.shape == VelocitySurface.LINE:
            z_intercept = VelocitySurface.XY_PLANE.intersection(Line(self.range))[0]
            self.__plot_point(z_intercept, 'black', '.')
            self.__plot_segment(Segment(z_intercept, self.range.p2), 'grey', '--', 0.5)
            self.__plot_segment(Segment(z_intercept, self.input_point), 'grey', '--', 0.5)
        if self.shape == VelocitySurface.SURFACE:
            self.ax.plot_wireframe(XI, YI, self.surface(XI, YI), rstride=10, cstride=10, color='grey', linewidth=0.25)

        self.__plot_segment(Segment(self.input_point, self.output_point), 'grey', '--', 0.5)
        self.__plot_point(self.output_point, 'red', 'o')
        plot.show(block=False)
        plot.pause(0.001)

    def get_velocity(self, point: Point):
        if not isinstance(point, Point): raise TypeError
        self.input_point = point
        if self.shape == VelocitySurface.SURFACE:
            self.output_point = Point(point.x, point.y, self.surface(point[0], point[1]).tolist()).evalf()
        elif self.shape == VelocitySurface.LINE:
            self.output_point = Line(self.range).projection(point)
        return self.output_point

    def __init__(self, *points):
        self.input_points = self.input_point = self.output_point = self.perp_seg = None
        self.scaled_edge_points = self.scaled_input = self.shape = self.surface = self.range = None
        self.x_limits = self.y_limits = None
        self.ax = plot.axes(projection="3d")
        self.input_points = points
        self.initialize(points)

    def initialize(self, points):
        if len(points) < 2: raise ValueError
        for pt in points:
            if not isinstance(pt, Point): raise TypeError

        if len(points) == 2:
            self.shape = VelocitySurface.LINE
        else:
            self.shape = VelocitySurface.SURFACE
            point_list = list(points)  # convert tuple to list

        scaled_points = [pt.scale(VelocitySurface.scale, VelocitySurface.scale, 1) for pt in point_list]
        segments = [Segment(pt, scaled_points[index+1]) for index, pt in enumerate((scaled_points + [scaled_points[0]])[:-1])]
        ss = self.__step_size(segments)
        edge_points = self.__edge_points(segments, ss, scaled_points[0], scaled_points[-1])

        self.range = segments[0]
        #
        # if self.shape == VelocitySurface.SURFACE:
        #     np.append(edge_points, scaled_points[0])
        #     # segments.append(Segment(scaled_points[-1], scaled_points[0]))  # close the figure
        # elif self.shape == VelocitySurface.LINE:
        #     self.range = segments[0]
        #     np.append(edge_points, scaled_points[-1])

        figure_points_array = np.array([scaled_points]).astype(float)
        self.scaled_input = [figure_points_array[:, 0], figure_points_array[:, 1], figure_points_array[:, 2]]
        self.scaled_edge_points = [edge_points[:, 0], edge_points[:, 1], edge_points[:, 2]]
        self.x_limits = [int(round(self.scaled_input[0].min(), 0)), int(round(self.scaled_input[0].max(), 0))]
        self.y_limits = [int(round(self.scaled_input[1].min(), 0)), int(round(self.scaled_input[1].max(), 0))]

        if self.shape == VelocitySurface.SURFACE:
            self.surface = RBF(self.scaled_edge_points[0], self.scaled_edge_points[1], self.scaled_edge_points[2], function='thin_plate', smooth=100.0)

p1 = Point(0, 0, 100)
p2 = Point(3, 0, 300)
p3 = Point(3, 4, 200)
p4 = Point(0, 4, 300)

vs = VelocitySurface(p1, p2, p3, p4)
# vs = VelocitySurface(p1, p3, p4)
# vs = VelocitySurface(p1, p3)

for i in range(1, 15):
    rand_pt = Point(randrange(max(vs.x_limits) - min(vs.x_limits)) + min(vs.x_limits), randrange(max(vs.y_limits) - min(vs.y_limits)) + min(vs.y_limits), 0)
    z = vs.get_velocity(rand_pt)
    vs.show_plot()
    sleep(2)
