import cv2
from matplotlib import pyplot

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

    def __init__(self, path):
        """Constructor that loads the image from a specified path and adds it as the first process on the list."""
        self._step_list = [Step(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), "base", {}, [])]

    def get_steps(self):
        """Returns the list of all steps."""
        return self._step_list

    def plot_all_steps(self):
        """Debug function for plotting all images in the process list."""
        fig, axes = pyplot.subplots(len(self._step_list), 1, figsize=(16, 3 * len(self._step_list)))
        for i in range(len(self._step_list)):
            if "binary" in self._step_list[i].attributes:
                axes[i].imshow(self._step_list[i].image, cmap='gray')
            else:
                axes[i].imshow(self._step_list[i].image)
            axes[i].axis('off')
            axes[i].set_title(self._step_list[i].step_name)
        pyplot.tight_layout()
        pyplot.show()

    def get_last_image(self):
        """Internal function used by process functions to get a working copy of the latest version of the image."""
        return self._step_list[-1].image.copy()

    def add_step(self, image, process_name, parameters, attributes):
        """Internal function used by process functions to add their result to the list."""
        self._step_list.append(Step(image, process_name, parameters, attributes))

    def _apply_step(self, step_function, parameters):
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

    def skeletonization(self, parameters=None):
        return self._apply_step(Transformations.skeletonization, parameters)

    def branch_removal(self, parameters=None):
        return self._apply_step(Transformations.branch_removal, parameters)
