import math
import tkinter as tk
import cv2
from PIL import Image, ImageDraw, ImageFont


class Graph:
    """
    Graph class using adjacency matrix representation
    """

    def __init__(self):
        self.vertices = []
        self.adjacency_matrix = [[]]
        self.shortest_path = []
        self.sample_graph()

    def add_vertex(self, x, y):
        """Adds a vertex to the graph at a given position on the image."""
        self.vertices.append((x, y))
        for row in self.adjacency_matrix:
            row.append(0)
        self.adjacency_matrix.append([0] * len(self.adjacency_matrix))

    def set_edge(self, u, v, weight, interpoints):
        """Sets the weight and interpolation points of an edge between two vertices."""
        self.adjacency_matrix[u][v] = (weight, interpoints)
        self.adjacency_matrix[v][u] = (weight, interpoints)

    def sample_graph(self):
        """Create a sample graph for testing purposes."""
        self.add_vertex(400, 100)
        self.add_vertex(300, 200)
        self.add_vertex(100, 300)
        self.add_vertex(200, 400)
        self.add_vertex(350, 300)

        self.set_edge(0, 1, 50, [(400, 100), (340, 130), (300, 200)])
        self.set_edge(1, 2, 75, [(300, 200), (250, 260), (170, 240), (100, 300)])
        self.set_edge(2, 3, 40, [(100, 300), (200, 400)])
        self.set_edge(3, 4, 60, [(200, 400), (350, 300)])
        self.set_edge(4, 1, 33, [(350, 300), (325, 250), (300, 200)])
        self.set_edge(1, 3, 125, [(300, 200), (260, 270), (220, 320), (200, 400)])

    def set_shortest_path(self, path):
        self.shortest_path = path

    def print_vertices(self):
        for i, vertex in enumerate(self.vertices):
            print("Vertex" + str(i) + ": " + str(vertex[0]) + " " + str(vertex[1]))

    def get_image_with_graph(self, image, font_size=20, with_shortest_path=False):
        draw = ImageDraw.Draw(image)
        self.draw_edges(draw, font_size, with_shortest_path)
        self.draw_vertices(draw, font_size, with_shortest_path)

        return image

    def draw_edges(self, draw, font_size=20, with_shortest_path=False):
        for i in range(0, len(self.adjacency_matrix)):
            for j in range(0, i):
                if self.adjacency_matrix[i][j] != 0:
                    points = self.adjacency_matrix[i][j][1]
                    for k in range(0, len(points) - 1):
                        if with_shortest_path and i in self.shortest_path and j in self.shortest_path and abs(
                                self.shortest_path.index(i) - self.shortest_path.index(j)) == 1:
                            draw.line((points[k][0], points[k][1], points[k + 1][0], points[k + 1][1]),
                                               fill="red", width=3)
                        else:
                            draw.line((points[k][0], points[k][1], points[k + 1][0], points[k + 1][1]),
                                               fill="black", width=3)

        def draw_text_in_rectangle(draw, myFont, points, text, line_width, margin):
            text_size = draw.textsize(text, myFont)
            text_position = (points[0] - text_size[0] / 2, points[1] - text_size[1] / 2)
            rectangle_position = [text_position, (text_position[0] + text_size[0] + 2 * margin,
                                                  text_position[1] + text_size[1])]
            draw.rectangle(rectangle_position, fill="white", outline='black', width=line_width)
            draw.text((text_position[0]+margin, text_position[1] - margin), text, font=myFont, fill='black')

        myFont = ImageFont.truetype('COMIC.ttf', font_size)
        line_width = min(max(int(font_size / 20 * 3), 1), 2)
        margin = int(font_size / 20 * 3)

        for i in range(0, len(self.adjacency_matrix)):
            for j in range(0, i):
                if self.adjacency_matrix[i][j] != 0:
                    points = self.adjacency_matrix[i][j][1]
                    if len(points) % 2 == 0:
                        ind1 = int(len(points) / 2) - 1
                        ind2 = int(len(points) / 2)
                        point = ((points[ind1][0] + points[ind2][0]) / 2,
                                 (points[ind1][1] + points[ind2][1]) / 2)
                    else:
                        ind = int(len(points) / 2)
                        point = points[ind]
                    draw_text_in_rectangle(draw, myFont, point,
                                                str(self.adjacency_matrix[i][j][0]),
                                                line_width, margin)

    def draw_vertices(self, draw, font_size=20, with_shortest_path=False):
        myFont = ImageFont.truetype('COMIC.ttf', font_size)

        for i, v in enumerate(self.vertices):
            text = str(i)
            text_width, text_height = draw.textsize(text, font=myFont)

            r = text_height / 2 + font_size / 4
            line_width = min(max(font_size // 10, 1), 2)

            if with_shortest_path and i in self.shortest_path:
                draw.ellipse((v[0] - r, v[1] - r, v[0] + r, v[1] + r), fill="red", outline="black", width=line_width)
            else:
                draw.ellipse((v[0] - r, v[1] - r, v[0] + r, v[1] + r), fill="lime", outline="black", width=line_width)

            text_x = v[0] - text_width / 2  # Centrowanie tekstu
            text_y = v[1] - text_height / 2 - font_size / 5
            draw.text((text_x, text_y), text, font=myFont, fill='black')




