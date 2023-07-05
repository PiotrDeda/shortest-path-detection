import cv2
import numpy as np

from graph import Graph
from transformations import Transformations


class Step:
    """
    Class for storing a raw image after a specific transformation was applied to the previous step.

    Members:
        image: raw image after the transformation was applied
        step_name: name of the transformation in the form of a string, e.g. "binarization", "morph_open", etc.
        parameters: dictionary of parameters used by the transformation, e.g. size of a morphological kernel
        attributes: meta-information about the image itself, currently the only one used is "binary" for binary images
    """

    def __init__(self, image, step_name, parameters, attributes):
        self.image = image
        self.step_name = step_name
        self.parameters = parameters
        self.attributes = attributes


class ProcImg:
    """
    Class for storing a processed image in the form of a list of all transformations (called "steps"),
    applied consecutively to the base image (which is always the first step in the list).
    """

    def __init__(self, image=None, image_path=None):
        """Constructor that loads the image from a specified path and adds it as the first process on the list."""
        self._step_list = [Step(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) if image_path
                                else image, "base", {}, [])]
        self._vertices = []
        self._edges = []
        self._graph = Graph()

    def get_steps(self):
        """Returns the list of all steps."""
        return self._step_list

    def get_vertices(self):
        """Returns the list of vertices."""
        return self._vertices

    def get_edges(self):
        """Returns the list of edges."""
        return self._edges

    def get_graph(self):
        """Returns the graph."""
        return self._graph

    def get_last_image(self):
        """Internal function used by process functions to get a working copy of the latest version of the image."""
        return self._step_list[-1].image.copy()

    def get_first_binary(self):
        """Internal function used by process functions to get a working copy of the latest image with the binary tag."""
        for step in self._step_list:
            if "binary" in step.attributes:
                return step.image.copy()

    def add_step(self, image, process_name, parameters, attributes):
        """Internal function used by process functions to add their result to the list."""
        self._step_list.append(Step(image, process_name, parameters, attributes))

    def add_vertices(self, vertices):
        """Internal function used by process functions to add vertices to the list."""
        self._vertices += vertices
        self._graph.add_vertices(vertices)

    def add_edge(self, edge):
        """Internal function used by process functions to add an edge to the list."""
        self._edges.append(edge)
        vertex_1_id = self._vertices.index(edge[0])
        vertex_2_id = self._vertices.index(edge[1])
        self._graph.set_edge(vertex_1_id, vertex_2_id, 1, edge[4])

    def set_weight_by_color(self, color, count):
        """Internal function used by process functions to set the weight of an edge by its color."""
        for edge in self._edges:
            if np.array_equal(edge[2], color):
                vertex_1_id = self._vertices.index(edge[0])
                vertex_2_id = self._vertices.index(edge[1])
                self._graph.set_edge(vertex_1_id, vertex_2_id, edge[3] * edge[3] / count, edge[4])

    def _apply_step(self, step_function, parameters):
        print(f"Applying {step_function.__name__}")
        if parameters is None:
            parameters = {}
        step_function(self, parameters)
        return self

    def segmentation(self, parameters=None):
        return self._apply_step(Transformations.segmentation, parameters)

    def binarization(self, parameters=None):
        return self._apply_step(Transformations.binarization, parameters)

    def morph_open(self, parameters=None):
        return self._apply_step(Transformations.morph_open, parameters)

    def morph_close(self, parameters=None):
        return self._apply_step(Transformations.morph_close, parameters)

    def filter(self, parameters=None):
        return self._apply_step(Transformations.filter, parameters)

    def skeletonization(self, parameters=None):
        return self._apply_step(Transformations.skeletonization, parameters)

    def branch_removal(self, parameters=None):
        return self._apply_step(Transformations.branch_removal, parameters)

    def vertex_search(self, parameters=None):
        return self._apply_step(Transformations.vertex_search, parameters)

    def vertex_deduplication(self, parameters=None):
        return self._apply_step(Transformations.vertex_deduplication, parameters)

    def path_coloring(self, parameters=None):
        return self._apply_step(Transformations.path_coloring, parameters)

    def path_flooding(self, parameters=None):
        return self._apply_step(Transformations.path_flooding, parameters)
