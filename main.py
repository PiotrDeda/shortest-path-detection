import os

from dijkstra_algorithm import dijkstra
from procimg import ProcImg


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
        sample_images[i].segmentation().binarization().morph_close().blur().skeletonization().branch_removal() \
            .vertex_search().vertex_deduplication().path_coloring().path_flooding().save_all_steps()

    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    # graph = Graph()
    # graph.sample_graph()
    graph = sample_images[0].get_graph()
    graph.draw_graph('maps/test6.png', "graphs/graph.png")

    shortest_distance, shortest_path = dijkstra(graph, 8, 31)

    print("Shortest distance:", shortest_distance)
    print("Shortest path:", shortest_path)

    graph.set_shortest_path(shortest_path)
    graph.draw_graph('maps/test6.png', "graphs/graph1.png")


if __name__ == '__main__':
    main()
