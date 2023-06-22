class Graph:
    """
    Graph class using adjacency matrix representation
    """

    def __init__(self):
        self.vertices = []
        self.adjacency_matrix = []

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

        self.set_edge(0, 1, 50, [(340, 130)])
        self.set_edge(1, 2, 75, [(250, 260), (170, 240)])
        self.set_edge(2, 3, 40, [])
        self.set_edge(3, 4, 60, [])
        self.set_edge(4, 1, 33, [(325, 250)])
        self.set_edge(1, 3, 125, [(260, 270), (220, 320)])
