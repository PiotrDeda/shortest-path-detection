import sys

def dijkstra(graph, source, destination):
    num_vertices = len(graph.vertices)

    distances = [sys.maxsize] * num_vertices

    distances[source] = 0

    visited = [False] * num_vertices

    previous = [-1] * num_vertices

    for _ in range(num_vertices):
        min_distance = sys.maxsize
        min_vertex = -1
        for v in range(num_vertices):
            if not visited[v] and distances[v] < min_distance:
                min_distance = distances[v]
                min_vertex = v

        visited[min_vertex] = True

        if min_vertex == destination:
            break

        for u in range(num_vertices):
            if graph.adjacency_matrix[min_vertex][u] == 0:
                weight = graph.adjacency_matrix[min_vertex][u]
            else:
                weight = graph.adjacency_matrix[min_vertex][u][0]
            if not visited[u] and weight > 0 and distances[min_vertex] + weight < distances[u]:
                distances[u] = distances[min_vertex] + weight
                previous[u] = min_vertex

    path = []
    current_vertex = destination
    while current_vertex != -1:
        path.append(current_vertex)
        current_vertex = previous[current_vertex]
    path.reverse()

    return distances[destination], path