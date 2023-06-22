from functools import wraps

import cv2
import numpy as np
from skimage.morphology import skeletonize


# noinspection PyIncorrectDocstring,PyUnresolvedReferences,PyUnusedLocal
class Transformations:
    """
    Class containing image transformation functions.
    To work with the ProcImg class, each function needs to have a signature (proc_img, parameters, image=None),
    where image is injected by the @_work_with_proc_img decorator. It should then be used as a regular image variable,
    together with parameters. The function should return a raw image, which the decorator automatically adds
    to the step list of the ProcImg.
    The first argument to the decorator is the name of the process, which will be stored in the step list.
    The second optional argument is a list of attributes which act as meta-information about the processed image;
    currently only "binary" is used to indicate binary images.
    """

    @staticmethod
    def _work_with_proc_img(process_name, attributes=None):
        if attributes is None:
            attributes = []

        def decorator(func):
            @wraps(func)
            def wrapper(proc_img, parameters, image=None):
                if image is None:
                    image = proc_img.get_last_image()
                result = func(proc_img, parameters, image)
                proc_img.add_step(result, process_name, parameters, attributes)

            return wrapper

        return decorator

    @staticmethod
    def _verify_parameters(required_parameters):
        def decorator(func):
            @wraps(func)
            def wrapper(proc_img, parameters, image=None):
                for (param, default) in required_parameters:
                    if param not in parameters:
                        parameters[param] = default
                return func(proc_img, parameters, image)

            return wrapper

        return decorator

    @staticmethod
    @_work_with_proc_img("segmentation")
    @_verify_parameters([("blur_kernel_shape", (3, 3)), ("k_means", 3)])
    def segmentation(proc_img, parameters, image=None):
        """
        Performs image segmentation using k-means clustering.

        Parameters:
            blur_kernel_shape (tuple): Kernel shape for Gaussian blur (default: (3, 3)).
            k_means (int): Number of clusters for k-means clustering (default: 3).

        """
        # Reshape the image into a 2D array of pixel values
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Define criteria for k-means clustering and perform function
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, parameters["k_means"],
                                        None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Assign the segmented colors based on the labels
        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]

        # Reshape image to input shape
        segmented_image = segmented_image.reshape(image.shape)

        # Apply Gaussian blur with given kernel shape
        out = cv2.GaussianBlur(segmented_image, parameters["blur_kernel_shape"], 0)

        return out

    @staticmethod
    @_work_with_proc_img("binarization", ["binary"])
    def binarization(proc_img, parameters, image=None):
        """
        Converts an input image to a binary image using Otsu's thresholding.

        Parameters:
            None
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    @staticmethod
    @_work_with_proc_img("morph_open", ["binary"])
    @_verify_parameters([("kernel_shape", (5, 5))])
    def morph_open(proc_img, parameters, image=None):
        """
        Performs morphological opening on a binary image.

        Parameters:
            kernel_shape (tuple): Kernel shape for morphological opening (default: (5, 5)).
        """
        kernel = np.ones(parameters["kernel_shape"], np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    @_work_with_proc_img("morph_close", ["binary"])
    @_verify_parameters([("kernel_shape", (3, 3))])
    def morph_close(proc_img, parameters, image=None):
        """
        Performs morphological closing on a binary image.

        Parameters:
            kernel_shape (tuple): Kernel shape for morphological closing (default: (3, 3)).
        """
        kernel = np.ones(parameters["kernel_shape"], np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    @_work_with_proc_img("skeletonization", ["binary"])
    def skeletonization(proc_img, parameters, image=None):
        """
        Performs skeletonization on a binary image.

        Parameters:
            None
        """
        img_skel = skeletonize(image / 255)
        return (img_skel.astype(np.uint8) ^ 1) * 255

    @staticmethod
    @_work_with_proc_img("branch_removal", ["binary"])
    @_verify_parameters([("tol", 0.05)])
    def branch_removal(proc_img, parameters, image=None):
        """
        Removes small branches from a skeletonized image.

        Parameters:
            tol (float): Threshold value to determine small branches (default: 0.05).
        """
        # Invert the skeletonized image
        img_skt_rev = cv2.bitwise_not(image)

        # Searching isolated elements on binary photo with connectedComponentsWithStats
        # connectivity = 4 -> two pixels are connected if are horizontally or vertically adjacent
        # connectivity = 8 -> same as with 4 but including corners
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_skt_rev, connectivity=8)

        # Calculating length of each element
        lengths = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]

        # Sorting all elements by length
        sorted_labels = sorted(range(1, num_labels), key=lambda x: lengths[x - 1], reverse=True)

        # To delete noises we keep only elements with length >= TOL
        longest_length = lengths[sorted_labels[0] - 1]
        num_components = sum(length >= parameters["tol"] * longest_length for length in lengths)

        # Creating mask with only the longest elements
        mask = np.zeros_like(image)
        for i in range(num_components):
            component_label = sorted_labels[i]
            mask[labels == component_label] = 255

        # Applying mask
        img_skt_filtered = cv2.bitwise_and(img_skt_rev, mask)
        return img_skt_filtered
