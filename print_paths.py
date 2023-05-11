from auxilary import construct_graph, print_graph
import matplotlib.pyplot as plt
import re
import os


def print_comparisons(comparisons, location):
    differing_parts = dict()
    for index in range(0, len(comparisons.keys()) - 1, 2):
        path1 = comparisons[list(comparisons.keys())[index]]
        path2 = comparisons[list(comparisons.keys())[index + 1]]
        indices = find_indices(path1, path2)
        name = list(comparisons.keys())[index]
        start, target = name.split("->")
        seed, size = target.split("(")[1].split(",")
        size = size[1]
        _, processed_graph = create_graph(16, 0)
        print_difference(
            processed_graph,
            path1,
            path2,
            indices,
            f"Seed 16 Size 0_Seed {seed} Size {size}_{start}_{target}",
            location,
            True,
        )
        print_difference(
            processed_graph,
            path1,
            path2,
            indices,
            f"Seed 16 Size 0_Seed {seed} Size {size}_{start}_{target}",
            location,
        )
    print(differing_parts)


def find_pair(all_paths, route_start, route_end, seed, comparisons):
    for path in all_paths:
        for path_name_2, path_2 in path.items():
            seed_2, size_2, route_start_2, route_end_2 = extract_details(path_name_2)
            if (
                route_start_2 == route_start
                and route_end == route_end_2
                and seed != seed_2
            ):
                route_2_cmp_2 = f"{route_start_2}->{route_end_2}({seed_2}, {size_2})"
                comparisons[route_2_cmp_2] = path_2


def read_paths(function_type, route_type):
    file_names = f"^{function_type}_.*_{route_type}.txt$"
    destination = f"./paths/{function_type}/"
    regex = re.compile(file_names)
    paths = []
    for root, dirs, files in os.walk(destination):
        for file in files:
            if regex.match(file):
                save_path(destination, paths, file)
    return paths


def save_path(destination, paths, file):
    with open(destination + file) as f:
        path = dict()
        path_list = list()
        for line in f.readlines():
            path_list.append(int(line.strip()))
        path[file] = path_list
        paths.append(path)
        f.close()


def create_graph(seed, size):
    graph = construct_graph("Andorra", size, seed)
    processed_graph = graph.get_pre_processed_graph()
    return graph, processed_graph


def extract_charging_stations(path, graph, processed_graph):
    charging_stations = [node for node in path if processed_graph.nodes[node]["CS"]]
    points = [
        (graph.graph.nodes[node]["x"], graph.graph.nodes[node]["y"])
        for node in charging_stations
    ]
    return points


def extract_details(path_name):
    file = path_name.split("_")
    seed, size = file[3:5]
    route_start, route_end = file[1:3]
    return int(seed), int(size), route_start, route_end


def print_paths(function_type, function_level, location):
    comparisons = dict()
    all_paths = read_paths(function_type, function_level)
    for path in all_paths:
        for path_name, path in path.items():
            seed, size, route_start, route_end = extract_details(path_name)
            graph, processed_graph = create_graph(seed, size)
            points = extract_charging_stations(path, graph, processed_graph)
            route = path
            if seed != 16 or size != 0:
                # highlight their differences
                route_2_cmp = f"{route_start}->{route_end}({seed}, {size})"
                comparisons[route_2_cmp] = path
                find_pair(all_paths, route_start, route_end, seed, comparisons)
            fig, ax = print_graph(processed_graph, "route", route)
            if points:
                a = [
                    ax.scatter(point[0], point[1], c="#001f3f", s=7**2, marker="*")
                    for point in points
                ][-1]
                plt.legend(
                    [a],
                    ["Charging station along route"],
                    scatterpoints=1,
                    loc="upper right",
                    ncol=3,
                    fontsize=14,
                )
            name = f"Route going from {route_start} to {route_end}"
            name += f" ({function_type}, {function_level}, {seed}, {size})"
            # fig.savefig(location + name + ".png")
    if comparisons:
        print_comparisons(comparisons, location)


def get_coordinates(graph, route, index):
    return (graph.nodes[route[index]]["x"], graph.nodes[route[index]]["y"])


def print_difference(graph, path1, path2, indices, name, location, zoom=False):
    # indices[0] is the diverging node
    path1_name, path2_name, start, target = name.split("_")
    left_shift_route = 1
    left_shift_graph = 2
    start_index_route = indices[0] - left_shift_route
    start_index_graph = indices[0] - left_shift_graph
    end_index_short_route = indices[1][1] + 1
    end_index_long_route = indices[1][0] + 1

    shorter_route = path1
    longer_route = path2
    shorter_route_name = path1_name
    longer_route_name = path2_name
    shorter_route_color = "#8A2BE2"
    longer_route_color = "#40E0D0"
    if len(path1) > len(path2):
        shorter_route = path2
        shorter_route_color = "#40E0D0"
        shorter_route_name = path2_name
        longer_route_name = path1_name
        longer_route = path1
        longer_route_color = "#8A2BE2"
    # subgraph on -1, the others only from indices
    subroute_path1 = shorter_route[start_index_route:end_index_short_route]
    subroute_path2 = longer_route[start_index_route:end_index_long_route]
    rc = [shorter_route_color, longer_route_color]
    # print the routes
    if zoom:
        # reverse one of them
        if subroute_path1[1] > shorter_route[start_index_route]:
            subroute_path1 = subroute_path1[::-1]
        if subroute_path2[1] > longer_route[start_index_route]:
            subroute_path2 = subroute_path2[::-1]
        graph = graph.subgraph(
            shorter_route[start_index_graph:end_index_short_route]
            + longer_route[start_index_graph:end_index_long_route]
        )
    routes = [subroute_path1, subroute_path2]
    fig, ax = print_graph(graph, "routes", routes, rc)
    if zoom:
        diverging_point = indices[0]
        joinig_point = indices[1][0]
        diverging_node = get_coordinates(graph, longer_route, diverging_point)
        joining_node = get_coordinates(graph, longer_route, joinig_point)
        shorter_route_node = get_coordinates(graph, shorter_route, start_index_route)
        longer_route_node = get_coordinates(graph, longer_route, start_index_route)
        a = ax.scatter(
            diverging_node[0],
            diverging_node[1],
            c="#001f3f",
            s=12**2 if zoom else 6**2,
            marker="o",
        )
        b = ax.scatter(
            joining_node[0],
            joining_node[1],
            c="#001f3f",
            s=12**2 if zoom else 6**2,
            marker="D",
        )
        c = ax.scatter(
            shorter_route_node[0],
            shorter_route_node[1],
            c=shorter_route_color,
            s=6**2,
            marker="o",
        )
        d = ax.scatter(
            longer_route_node[0],
            longer_route_node[1],
            c=longer_route_color,
            s=6**2,
            marker="o",
        )
        plt.legend(
            [a, b, c, d],
            ["Diverging node", "Joining node", shorter_route_name, longer_route_name],
            scatterpoints=1,
            fontsize=14,
        )
        name += "zoomed_in"
    # fig.savefig(location + name + ".png")


def find_indices(path1, path2):
    differing_indices = []
    if len(path1) < len(path2):
        for path_index, node in enumerate(path1):
            if path2[path_index] != node:
                differing_indices.append(path_index)
                break
        longer_path_index, shorter_path_index = find_link(
            path2, path1, differing_indices[0]
        )
    else:
        for path_index, node in enumerate(path2):
            if path1[path_index] != node:
                differing_indices.append(path_index)
                break
        longer_path_index, shorter_path_index = find_link(
            path1, path2, differing_indices[0]
        )
    differing_indices.append((longer_path_index, shorter_path_index))
    return differing_indices

    # find where they link again


def find_link(longer_path, shorter_path, index):
    for index_, node in enumerate(longer_path[index:]):
        try:
            path2_index = shorter_path[index:].index(node) + index
            return (index_ + index, path2_index)
        except:
            continue


if __name__ == "__main__":
    location = "./data/pics/version2/"
    print_paths("bis4ev", "hard", location)
    print_paths("bis4ev", "easy", location)
