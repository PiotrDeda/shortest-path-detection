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

    def draw_graph(self, path, output):
        img = Image.open(path)
        width = img.width
        height = img.height

        draw = ImageDraw.Draw(img)
        self.draw_edges(draw)
        self.draw_vertices(draw, width)
        img.save(output)

    def draw_edges(self, draw):
        for i in range(0, len(self.adjacency_matrix)):
            for j in range(0, i):
                if self.adjacency_matrix[i][j] != 0:
                    points = self.adjacency_matrix[i][j][1]
                    for k in range(0, len(points) - 1):
                        if i in self.shortest_path and j in self.shortest_path and abs(
                                self.shortest_path.index(i) - self.shortest_path.index(j)) == 1:
                            draw.line((points[k][0], points[k][1], points[k + 1][0], points[k + 1][1]),
                                               fill="red", width=3)
                        else:
                            draw.line((points[k][0], points[k][1], points[k + 1][0], points[k + 1][1]),
                                               fill="black", width=3)

        for i in range(0, len(self.adjacency_matrix)):
            for j in range(0, i):
                if self.adjacency_matrix[i][j] != 0:
                    points = self.adjacency_matrix[i][j][1]
                    if len(points) % 2 == 0:
                        ind1 = int(len(points) / 2) - 1
                        ind2 = int(len(points) / 2)
                        myFont = ImageFont.truetype('COMIC.ttf', 20)
                        text_size = draw.textsize(str(self.adjacency_matrix[i][j][0]), myFont)
                        text_position = ((points[ind1][0] + points[ind2][0]) / 2 - text_size[0]/2,(points[ind1][1] + points[ind2][1]) / 2 - text_size[1]/2)
                        rectangle_position = [text_position, (text_position[0] + text_size[0], text_position[1] + text_size[1])]
                        draw.rectangle(rectangle_position, fill="white", outline='black')
                        draw.text(((points[ind1][0] + points[ind2][0]) / 2 - text_size[0]/2,
                                                 (points[ind1][1] + points[ind2][1]) / 2 - text_size[1]/2 - 3),
                                                 str(self.adjacency_matrix[i][j][0]),
                                                 font=myFont, fill='black')
                    else:
                        ind = int(len(points) / 2)
                        myFont = ImageFont.truetype('COMIC.ttf', 20)
                        text_size = draw.textsize(str(self.adjacency_matrix[i][j][0]), myFont)
                        text_position = (points[ind][0] - text_size[0]/2, points[ind][1] - text_size[1]/2)
                        rectangle_position = [text_position, (text_position[0] + text_size[0], text_position[1] + text_size[1])]
                        draw.rectangle(rectangle_position, fill="white", outline='black')
                        draw.text((points[ind][0] - text_size[0]/2, points[ind][1] - text_size[1]/2 - 3),
                                                 str(self.adjacency_matrix[i][j][0]),
                                                 font=myFont, fill='black')

    def draw_vertices(self, draw, width):
        center_x = width / 2
        R = center_x * 1 / 6
        n = len(self.vertices)
        if n > 3:
            r = R / n
        else:
            r = R / n * 0.5
        for i, v in enumerate(self.vertices):
            if i in self.shortest_path:
                draw.ellipse((v[0] - r, v[1] - r, v[0] + r, v[1] + r), fill="red", outline="black", width=3)
            else:
                draw.ellipse((v[0] - r, v[1] - r, v[0] + r, v[1] + r), fill="lime", outline="black", width=3)
            myFont = ImageFont.truetype('COMIC.ttf', 20)
            draw.text((v[0] - r/2, v[1] - r,), str(i), font=myFont, fill='black')

