"""
Testing Suite
========

The testing suite for this project
"""

import time
from typing import List, Tuple
from auxilary import (
    calculate_charging_time_bis4ev,
    calculate_charging_time_dijkstra,
    calculate_travel_time,
    construct_graph,
    construct_contraction_graph,
    expand_path,
    get_path_metric,
    print_route,
    read_paths,
)
from BiS4EV import bis_4_ev
from pandas import DataFrame as DF
import networkx as nx
from Dijkstra4EV import Dijkstra4EV_route_planner


class Testing_Suite_Error(Exception):
    pass


# Global variables
places_dict = {
    "Andorra La Vella": 2758504422,
    "Escaldes-Engordany": 51974050,
    "Sant Julià de Lòria": 3610820021,
    "Encamp": 1934179573,
    "La Massana": 2975949958,
    "Ordino": 9721876525,
    "Canillo": 53275523,
    "El Pas de la Casa": 7296679289, 
    "Arinsal": 52204280,
    "La Margineda": 277697480,
    "Sispony": 3268159387,
    "Llorts": 268615705,
    "Erts": 51554703,
    "Bixessarri": 52677211,
    "Aubinyà": 52262417,
    "El Serrat": 51582448,
}
columns = [
    "seed",
    "size",
    "runtime",
    "total_travel_time",
    "total_energy_cost",
    "total_distance",
    "total_charging_time",
    "outcome",
    "type",
    "route",
]
hard_routes = [("El Pas de la Casa", "Arinsal"), ("El Serrat", "Aubinyà")]
easy_routes = [
    ("Encamp", "Canillo"),
    ("Ordino", "La Massana"),
    ("Bixessarri", "Sant Julià de Lòria"),
    ("Arinsal", "Ordino"),
    ("Escaldes-Engordany", "Encamp"),
    ("Andorra La Vella", "Sispony"),
    ("La Margineda", "Aubinyà"),
    ("Llorts", "Ordino"),
    ("Canillo", "Ordino"),
    ("Andorra La Vella", "Erts"),
]

# TODO: fix ShapelyDeprecationWarning


def save_route(path: str, folder: str, filename: str):
    """saves the given path/route to the given folder in the current directories paths subdirectory

    Args:
        path (str)
        folder (str)
        filename (str)
    """
    save_path = f"./paths/{folder}/{filename}.txt"
    with open(save_path, "w") as f:
        for line in path:
            f.write(f"{line}\n")
    f.close()


def get_graph(
    seed: int, nbr_charging_stations: int = 1, location: str = "Andorra"
) -> nx.MultiGraph:
    """returns the processed graph for the given conditions

    Args:
        seed (int)
        nbr_charging_stations (int, optional). Defaults to 1.
        location (str, optional). Defaults to 'Andorra'.

    Returns:
        nx.MultiGraph
    """
    graph = construct_graph(location, nbr_charging_stations, seed, places_dict.values())
    processed_graph = graph.get_pre_processed_graph()
    return processed_graph


def get_test_data():
    all_paths = []
    run_test_suite(
        function_type="bis4ev", route_type="easy", iteration_nbr=10, paths=all_paths
    )
    if not all_paths:
        all_paths = read_paths("bis4ev", "easy")
    run_test_suite(
        function_type="dijkstra", route_type="easy", iteration_nbr=10, paths=all_paths
    )


def run_test_suite(
    function_type: str, route_type: str, iteration_nbr: int, paths: List[List[int]]
):
    """runs the tests according to the given parameters

    Saves the data to the data subdirectory of the current directory

    Args:
        function_type (str)
        route_type (str)
        iteration_nbr (int)
        paths (List[List[int]])
    """
    seeds = [16, 1963, 29, 60, 4]
    test_sizes = [0, 3, 5]
    dataframe = DF(columns=columns)
    for index, seed in enumerate(seeds):
        for size_index, times in enumerate(test_sizes):
            processed_graph = get_graph(seed, times)
            start_index = extract_start_index(index, size_index, test_sizes, route_type)
            routes = extract_routes(route_type)
            test_routes(
                processed_graph,
                iteration_nbr,
                dataframe,
                routes,
                route_type,
                start_index,
                paths,
                function_type,
                seed,
                times,
            )
        dataframe.to_excel(
            "./data/" + function_type + "/" + route_type + "_" + str(seed) + ".xlsx",
            index=False,
        )
    dataframe.to_excel(
        "./data/" + function_type + "/" + route_type + ".xlsx", index=False
    )
    print("Tests have concluded.")


def extract_start_index(
    index: int, size_index: int, test_sizes: int, route_type: str
) -> int:
    """function to determine the start index of where the dataframe should be updated with data next for the current iterations
      based on the previous iterations and conditions specified when running run_test_suite

    Args:
        index (int)
        size_index (int)
        test_sizes (int)
        route_type (str)

    Raises:
        Testing_Suite_Error: if the route type is not 'easy' or 'hard' the exception is thrown

    Returns:
        int
    """
    route_nbr = None
    if route_type == "easy":
        route_nbr = len(easy_routes)
    elif route_type == "hard":
        route_nbr = len(hard_routes)
    elif route_type == "both":
        route_nbr = len(easy_routes) + len(hard_routes)
    else:
        raise Testing_Suite_Error("Invalid route_type")
    return (size_index + index) * route_nbr + index * route_nbr * len(test_sizes)


def extract_routes(route_type: str) -> List[Tuple[str, str]]:
    """function to return the list of routes based on the type specified

    Args:
        route_type (str)

    Raises:
        Testing_Suite_Error: if the route type is not 'easy' or 'hard' the exception is thrown

    Returns:
        List[Tuple[str,str]]
    """
    if route_type == "easy":
        routes = easy_routes
    elif route_type == "hard":
        routes = hard_routes
    elif route_type == "both":
        routes = easy_routes + hard_routes
    else:
        raise Testing_Suite_Error("Invalid route_type")
    return routes


def test_routes(
    processed_graph: nx.MultiGraph,
    no_of_iterations: int,
    dataframe: DF,
    routes: List[Tuple[str, str]],
    type: str,
    start_index: int,
    paths: List[List[int]],
    function_type: str,
    seed: int,
    size: int,
):
    for index, route in enumerate(routes):
        print(f"{function_type} testing {route} ({type})")
        times = []
        total_travel_times = []
        total_energy_costs = []
        total_distances = []
        total_charging_times = []
        start = places_dict[route[0]]
        target = places_dict[route[1]]
        flag = run_iterations(
            processed_graph,
            no_of_iterations,
            function_type,
            times,
            total_travel_times,
            total_energy_costs,
            total_distances,
            total_charging_times,
            start,
            target,
            paths,
            route,
            seed,
            size,
            type,
        )
        update_dataframe(
            dataframe,
            type,
            start_index + index,
            route,
            times,
            total_travel_times,
            total_energy_costs,
            total_distances,
            total_charging_times,
            flag,
            seed,
            size,
        )


def update_dataframe(
    df: DF,
    type: str,
    insert_index: int,
    route: Tuple[str, str],
    times: List[float],
    total_travel_times: List[float],
    total_energy_costs: List[float],
    total_distances: List[float],
    total_charging_times: List[float],
    flag: str,
    seed: int,
    size: int,
):
    """updates the dataframe based on the data collected during the iterations

    Args:
        df (DF)
        type (str)
        insert_index (int)
        route (Tuple[str, str])
        times (List[float])
        total_travel_times (List[float])
        total_energy_costs (List[float])
        total_distances (List[float])
        total_charging_times (List[float])
        flag (str)
        seed (int)
        size (int)
    """
    df.at[insert_index, "seed"] = seed
    df.at[insert_index, "size"] = size
    df.at[insert_index, "type"] = type
    df.at[insert_index, "route"] = route
    df.at[insert_index, "outcome"] = flag
    df.at[insert_index, "runtime"] = sum(times) / len(times)
    add_metric(df, insert_index, "total_travel_time", total_travel_times)
    add_metric(df, insert_index, "total_energy_cost", total_energy_costs)
    add_metric(df, insert_index, "total_distance", total_distances)
    add_metric(df, insert_index, "total_charging_time", total_charging_times)


def run_iterations(
    processed_graph: nx.MultiGraph,
    no_of_iterations: int,
    function_type: str,
    times: List[float],
    total_travel_times: List[float],
    total_energy_costs: List[float],
    total_distances: List[float],
    total_charging_times: List[float],
    start: int,
    target: int,
    paths: List[List[int]],
    route,
    seed: int,
    size: int,
    type: str,
) -> str:
    """runs the iteration for the conditions for the route

    Args:
        processed_graph (nx.MultiGraph)
        no_of_iterations (int)
        function_type (str)
        times (List[float])
        total_travel_times (List[float])
        total_energy_costs (List[float])
        total_distances (List[float])
        total_charging_times (List[float])
        start (int)
        target (int)
        paths (List[List[int]])
        route (_type_)
        seed (int)
        size (int)
        type (str)

    Returns:
        str
    """
    for _ in range(no_of_iterations):
        contraction_time = 0
        if function_type == "bis4ev":
            t1 = time.time()
            contracted_graph = construct_contraction_graph(processed_graph)
            t2 = time.time()
            contraction_time = t2 - t1
        # let the run time be comparing when they have a route that can be printed
        t1 = time.time()
        if function_type == "dijkstra":
            # get the path
            path, charging_stops = Dijkstra4EV_route_planner(
                processed_graph, start, target
            )
        elif function_type == "bis4ev":
            # get the path
            planner = bis_4_ev(contracted_graph, start, target)
            path, charging_stops = planner.find_route()
            # expand it and then use it to find stuff
            if path != -1:
                path = expand_path(contracted_graph, processed_graph, path)
        t2 = time.time()
        iteration_time = (t2 - t1) + contraction_time
        times.append(iteration_time)
        if path != -1:
            total_time, total_energy, distance, charge_time = extract_metrics(
                processed_graph, path, function_type, charging_stops
            )
            if path not in paths:
                # only storing unique paths across one function type
                paths.append(path)
                print_route(
                    processed_graph,
                    path,
                    "_".join(
                        [function_type, route[0], route[1], str(seed), str(size), type]
                    ),
                )
                save_route(
                    path,
                    function_type,
                    "_".join(
                        [function_type, route[0], route[1], str(seed), str(size), type]
                    ),
                )
            total_travel_times.append(total_time)
            total_energy_costs.append(total_energy)
            total_distances.append(distance)
            total_charging_times.append(charge_time)
        else:
            return "Failed"
        if times[-1] > 300:
            return "Capped"
    return "Passed"


def extract_metrics(
    processed_graph: nx.MultiGraph,
    path: List[int],
    function_type: str,
    charging_stops: List[int],
) -> Tuple[float, float, float, float]:
    """function to extract the metrics from a finished iteration

    Args:
        processed_graph (nx.MultiGraph)
        path (List[int])
        function_type (str)
        charging_stops (List[int])

    Returns:
        Tuple[float, float, float, float]
    """
    total_time = 0.0
    charge_time = 0.0
    total_time += calculate_travel_time(processed_graph, path)
    if function_type == "dijkstra":
        charge_time = calculate_charging_time_dijkstra(charging_stops)
    elif function_type == "bis4ev":
        # would find charging flags because of the same graph
        # want to use the original graph but include the flags needed for charging policy
        charge_time = calculate_charging_time_bis4ev(
            processed_graph, path, charging_stops
        )
    total_time += charge_time
    total_energy = get_path_metric(path, processed_graph, "battery_consumption")
    distance = get_path_metric(path, processed_graph, "length")
    return total_time, total_energy, distance, charge_time


def add_metric(df: DF, index: int, label: str, values: List[float]):
    """function to update a given dataframe at a given index for a given label with given values

    Args:
        df (DF)
        index (int)
        label (str)
        values (List[float])
    """
    try:
        df.at[index, label] = sum(values) / len(values)
    except ZeroDivisionError:
        df.at[index, label] = -1


if __name__ == "__main__":
    get_test_data()
