from functools import wraps

import cv2
import numpy as np
from skimage.morphology import thin


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
    def _color_generator():
        """
        Generates consecutive RGB colors.
        Excludes [0, 0, 0] and any color with a value of 255 (reserved special colors).
        """
        for r in range(1, 255):
            for g in range(0, 255):
                for b in range(0, 255):
                    yield [r, g, b]

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
    @_work_with_proc_img("filter", ["binary"])
    def filter(proc_img, parameters, image=None):
        """
        Applies bilateral filter to an image.

        Parameters:
            None
        """
        return cv2.bilateralFilter(image, 20, 75, 75)

    @staticmethod
    @_work_with_proc_img("skeletonization", ["binary"])
    def skeletonization(proc_img, parameters, image=None):
        """
        Performs skeletonization on a binary image.

        Parameters:
            None
        """
        img_skel = thin(image / 255)
        return cv2.bitwise_not((img_skel.astype(np.uint8) ^ 1) * 255)

    @staticmethod
    @_work_with_proc_img("branch_removal", ["binary"])
    @_verify_parameters([("tol", 0.05)])
    def branch_removal(proc_img, parameters, image=None):
        """
        Removes small branches from a skeletonized image.

        Parameters:
            tol (float): Threshold value to determine small branches (default: 0.05).
        """
        # Searching isolated elements on binary photo with connectedComponentsWithStats
        # connectivity = 4 -> two pixels are connected if are horizontally or vertically adjacent
        # connectivity = 8 -> same as with 4 but including corners
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

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
        img_skt_filtered = cv2.bitwise_and(image, mask)
        return img_skt_filtered

    @staticmethod
    @_work_with_proc_img("vertex_search")
    def vertex_search(proc_img, parameters, image=None):
        """
        Finds junction points in a skeletonized image.

        Parameters:
            None
        """
        # Transform image representation from grayscale to BGR and make a working copy
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result = image.copy()

        # For each filled pixel in the image...
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 0] == 255:
                    # ...handle edges...
                    if i == 0 or j == 0 or i == image.shape[0] - 1 or j == image.shape[1] - 1:
                        if i == 0:
                            neighbor_sum = np.sum(image[i:i + 2, j - 1:j + 2, 0]) - 255
                        elif i == image.shape[0] - 1:
                            neighbor_sum = np.sum(image[i - 1:i + 1, j - 1:j + 2, 0]) - 255
                        elif j == 0:
                            neighbor_sum = np.sum(image[i - 1:i + 2, j:j + 2, 0]) - 255
                        else:
                            neighbor_sum = np.sum(image[i - 1:i + 2, j - 1:j + 1, 0]) - 255
                    # ...and calculate sum of values of surrounding pixels
                    else:
                        neighbor_sum = np.sum(image[i - 1:i + 2, j - 1:j + 2, 0]) - 255

                    # If there are two filled pixels, it's a path, otherwise it's a junction point
                    if neighbor_sum == 255 * 2:
                        result[i, j] = [255, 0, 0]
                    else:
                        result[i, j] = [0, 255, 0]

        return result

    @staticmethod
    @_work_with_proc_img("vertex_deduplication")
    def vertex_deduplication(proc_img, parameters, image=None):
        """
        Removes erroneous adjacent junction points from an image.

        Parameters:
            None
        """
        # For each green pixel in the image...
        (width, height) = image.shape[:2]
        for i in range(width):
            for j in range(height):
                if image[i, j, 1] == 255:
                    # ...handle edges...
                    if i == 0 or j == 0 or i == width - 1 or j == height - 1:
                        if i == 0:
                            neighbors = (np.array([i, i + 1, i]), np.array([j - 1, j, j + 1]))
                        elif i == image.shape[0] - 1:
                            neighbors = (np.array([i - 1, i, i]), np.array([j - 1, j, j + 1]))
                        elif j == 0:
                            neighbors = (np.array([i - 1, i + 1, i]), np.array([j, j, j + 1]))
                        else:
                            neighbors = (np.array([i - 1, i + 1, i]), np.array([j - 1, j, j]))
                    # ...and calculate sum of cardinally adjacent green pixels
                    else:
                        neighbors = (np.array([i - 1, i + 1, i, i]), np.array([j, j, j - 1, j + 1]))
                    # If there are two or more green pixels, turn them blue
                    if np.sum(image[neighbors], axis=0)[1] >= 2 * 255:
                        for k in range(len(neighbors[0])):
                            if image[neighbors[0][k], neighbors[1][k], 1] == 255:
                                image[neighbors[0][k], neighbors[1][k]] = [255, 0, 0]

        # Add each green pixel into a list of vertices
        green_vertices = np.argwhere(image[:, :, 1] == 255)
        proc_img.add_vertices([(j, i) for i, j in green_vertices])

        return image

    @staticmethod
    @_work_with_proc_img("path_coloring")
    def path_coloring(proc_img, parameters, image=None):
        """
        Colors the paths in an image, each with a different color.

        Parameters:
            None
        """
        # Initialize generator for consecutive colors that will uniquely identify paths
        color_generator = Transformations._color_generator()

        # For each vertex...
        for vertex in proc_img.get_vertices():
            # ...try to fire a path in all 8 directions
            for i in range(vertex[1] - 1, vertex[1] + 2):
                for j in range(vertex[0] - 1, vertex[0] + 2):
                    # Handle edges
                    if i < 0 or j < 0 or i >= image.shape[0] or j >= image.shape[1]:
                        continue

                    # Pointer marks current position, shadow marks previous position
                    pointer = np.array([i, j])
                    shadow = np.array([vertex[1], vertex[0]])

                    # Only set color when a path can be started
                    color_set = False
                    color = None

                    # Initialize path length (with 1 to prevent division by 0)
                    length = 1

                    # Add vertex as the first intermediate point of the path
                    inter_points = [tuple(vertex)]

                    # Try to continue the path until it reaches a vertex or can't paint
                    while True:
                        # If pixel at pointer is blue, paint it
                        if image[pointer[0], pointer[1], 0] == 255:
                            if not color_set:
                                color = next(color_generator)
                                color_set = True
                            length += 1
                            # Add an intermediate point every 10 pixels
                            if length % 10 == 0:
                                inter_points.append((pointer[1], pointer[0]))
                            image[pointer[0], pointer[1]] = color
                        # Handle case where vertices are adjacent
                        elif image[pointer[0], pointer[1], 1] == 255:
                            if not color_set:
                                color = next(color_generator)
                                color_set = True
                            inter_points.append((pointer[1], pointer[0]))
                            proc_img.add_edge(((vertex[0], vertex[1]), (pointer[1], pointer[0]),
                                               color, length, inter_points))
                            break
                        # If not possible to start the path, break
                        else:
                            break

                        # Search adjacent pixels for a vertex
                        found_vertex = False
                        for k in range(pointer[0] - 1, pointer[0] + 2):
                            for m in range(pointer[1] - 1, pointer[1] + 2):
                                # If pointer is at vertex other than the starting one, finish the path and add as edge
                                if 0 <= k < image.shape[0] and 0 <= m < image.shape[1] \
                                        and image[k, m, 1] == 255 and not np.array_equal(np.array([k, m]), shadow):
                                    found_vertex = True
                                    if not color_set:
                                        color = next(color_generator)
                                        color_set = True
                                    inter_points.append((m, k))
                                    proc_img.add_edge(((vertex[0], vertex[1]), (m, k), color, length, inter_points))
                                    break
                            else:
                                continue
                            break
                        if found_vertex:
                            break

                        # Search adjacent pixels for path continuation
                        for k in range(pointer[0] - 1, pointer[0] + 2):
                            for m in range(pointer[1] - 1, pointer[1] + 2):
                                if 0 <= k < image.shape[0] and 0 <= m < image.shape[1] \
                                        and image[k, m, 0] == 255 and not np.array_equal(np.array([k, m]), shadow) \
                                        and not (vertex[1] - 1 <= k <= vertex[1] + 1 and
                                                 vertex[0] - 1 <= m <= vertex[0] + 1):
                                    shadow = pointer
                                    pointer = np.array([k, m])
                                    break
                            else:
                                continue
                            break

        return image

    @staticmethod
    @_work_with_proc_img("path_flooding")
    def path_flooding(proc_img, parameters, image=None):
        """
        Floods the paths in an image with their respective colors and calculates path weights.

        Parameters:
            None
        """
        # Get the binary image which will be used as a reference for which pixels need to be flooded
        binary_background = proc_img.get_first_binary()

        # Find all colored path pixels
        active_pixels = np.argwhere(np.logical_and(np.any(image != [0, 0, 0], axis=2), image[:, :, 1] != 255))

        # Initialize optimization variables
        (width, height) = image.shape[:2]
        black = np.array([0, 0, 0])

        # Run loop until no changes were made
        running = True
        while running:
            running = False

            # Initialize optimization variables
            new_active_pixels = []
            append = new_active_pixels.append

            # For each active pixel, paint adjacent pixels that are filled in the binary background with the same color
            for (i, j) in active_pixels:
                for k in range(i - 1, i + 2):
                    for m in range(j - 1, j + 2):
                        if 0 <= k < width and 0 <= m < height and binary_background[k, m] == 255 \
                                and np.array_equal(image[k, m], black):
                            image[k, m] = image[i, j]
                            append((k, m))
                            running = True

            # Make the pixels painted this loop the active pixels for the next loop
            active_pixels = new_active_pixels

        # Count the number of pixels of each color and set the path weights accordingly
        # The formula for weight is (width of path) ^ 2 / (number of pixels)
        colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
        for color, count in zip(colors, counts):
            proc_img.set_weight_by_color(color, count)

        return image
