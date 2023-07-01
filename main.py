from procimg import ProcImg
from graph import Graph
from dijkstra_algorithm import dijkstra


def main():
    sample_images = [
        ProcImg("maps/test1.jpg"),
        ProcImg("maps/test2.png"),
        ProcImg("maps/test3.png"),
        ProcImg("maps/test4.png"),
        ProcImg("maps/test5.jpg"),
        ProcImg("maps/test6.png"),
    ]

    for i in range(len(sample_images)):
        sample_images[i].segmentation().binarization().morph_close().skeletonization().branch_removal().plot_all_steps()

    graph = Graph()
    graph.sample_graph()
    graph.draw_graph('maps/test3.png', "graphs/graph.png")

    shortest_distance, shortest_path = dijkstra(graph, 3, 1)

    print("Najkrótsza odległość:", shortest_distance)
    print("Najkrótsza ścieżka:", shortest_path)

    graph.set_shortest_path(shortest_path)
    graph.draw_graph('maps/test3.png', "graphs/graph1.png")

if __name__ == '__main__':
    main()
