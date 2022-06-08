import math

from .SetBasedAbstraction import SetBasedAbstraction
from abstractions.sets.Box import Box
from utils.Options import *


class BoxAbstraction(SetBasedAbstraction):
    def __init__(self, confidence_fun, size=1, epsilon=0., epsilon_relative=USE_EPSILON_RELATIVE):
        super().__init__(confidence_fun, size, epsilon, epsilon_relative)

    def name(self):
        return "Box"

    def set_type(self):
        return Box

    def distance_to_boxes(self, box_abstraction):
        distances = []
        for box in self.sets:
            d = box.distance_to_boxes(box_abstraction.sets)
            distances.append(d)
        return distances

    def box_distance(self, point):
        return min(box.box_distance(point) for box in self.sets)

    def closest_box_with_box_distance(self, point):
        d = math.inf
        b = None
        for box in self.sets:
            d_new = box.box_distance(point)
            if d_new < d:
                d = d_new
                b = box
        return b

    def plot_centers(self, dims, ax, color, marker):
        for box in self.sets:
            box.plot_center(dims, ax, color, marker)
