"""
Dijkstra4EV
========


Dijkstra4EV is a routing algorithm for electric vehicle. 

The code here is the implementation of Dijkstra4EV for this project
"""

from typing import Dict, List, Tuple, Union
import networkx as nx
import heapq

from auxilary import get_edge_metric, node_is_cs


class Dijkstra4EV_Exception(Exception):
    pass


def Dijkstra4EV_route_planner(
    graph: nx.MultiGraph, start: int, target: int
) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
    """a function that plans a route using Dijkstra4EV

    Args:
        graph (nx.MultiGraph)
        start (int)
        target (int)

    Raises:
        Dijkstra4EV_Exception: if the graph has not been processed, raises an error

    Returns:
        Union[Tuple[int,int], Tuple[List[int], List[int]]]: Either returns a flag saying no route was found, or the route found and
        the charging stops along the way
    """
    if graph.name != "processed":
        raise Dijkstra4EV_Exception("Graph must be processed")
    paths = list()
    battery_consumption_table = {node: float("inf") for node in graph.nodes()}
    battery_consumption_table[start] = 0
    battery_consumption_previous_stopover_node_table = {
        node: None for node in graph.nodes()
    }
    visited = set()
    Dijkstra4EV(
        graph=graph,
        start=start,
        end=target,
        battery_consumption_table=battery_consumption_table,
        battery_consumption_previous_stopover_node_table=battery_consumption_previous_stopover_node_table,
        paths=paths,
        visited=visited,
    )
    paths = [construct_path(start, target, path_version) for path_version in paths]
    if len(paths):
        path = paths[-1]
        charging_stops = find_charging_stops(graph, path)
        return path, charging_stops  # , len(visited)
    else:
        return -1, -1


def find_charging_stops(
    graph: nx.MultiGraph, path: List[int]
) -> List[Tuple[int, Union[int, float]]]:
    """A function that finds where all the charging stops should be along the given path sed the charging policy developed for Dijkstra4EV

    Args:
        graph (nx.MultiGraph)
        path (List[int])

    Returns:
        List[Tuple[int, Union[int, float]]]
    """
    metric = 0
    charging_stops = []
    for index in range(len(path) - 1):
        node1 = path[index]
        node2 = path[index + 1]
        metric += get_edge_metric(graph, node1, node2, "battery_consumption")
        if metric <= 90 and node_is_cs(graph, node2):
            charging_stops.append((node1, 100 - metric))  #  charge here
            metric = 0
    return charging_stops


def Dijkstra4EV(
    graph: nx.MultiGraph,
    start: int,
    end: int,
    battery_consumption_table: Dict[str, float],
    battery_consumption_previous_stopover_node_table: Dict[int, int],
    paths: List[Dict[str, int]] = [],
    initial_cost: int = 0,
    visited=set(),
):
    """The path finding code for Dijkstra4EV

    Args:
        graph (nx.MultiGraph)
        start (int)
        end (int)
        battery_consumption_table (Dict[str, int])
        battery_consumption_previous_stopover_node_table (Dict[int, int])
        paths (List[Dict[str, int]], optional). Defaults to [].
        initial_cost (int, optional). Defaults to 0.
        visited (_type_, optional). Defaults to set().

    Returns:
        bool
    """
    priority_queue = [(initial_cost, start)]
    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)
        if current_node == end:
            print("found end")
            paths.append(battery_consumption_previous_stopover_node_table.copy())
        if current_node in visited:
            continue
        # only added if it is added to the priority_queue which only happens if that way is the best
        # but if that way doesn't work, that way should be closed
        # and opened up for those which were not ideal but that might work
        visited.add(current_node)
        # iterate over neighbours/adjacent nodes
        for edge in graph.edges(current_node):
            neighbor = edge[1]
            if neighbor in visited:
                continue
            # weight = graph.get_edge_data(edge[0], neighbor)[0]['battery_consumption']
            weight = get_edge_metric(graph, current_node, neighbor)
            # cost of going from the start node to the neighbour using the current_node as the second to last stop
            cost_so_far = battery_consumption_table[current_node] + weight
            cost_to_neighbour_through_current_node = (
                cost_so_far  # cost of getting to current_node to neighbour
            )
            # if this is the cheapest way so far and we can get there :: only adds reachable and smaller
            if (
                cost_to_neighbour_through_current_node
                < battery_consumption_table[neighbor]
                and weight < 90
            ):
                battery_consumption_table[
                    neighbor
                ] = cost_to_neighbour_through_current_node
                battery_consumption_previous_stopover_node_table[
                    neighbor
                ] = current_node
                outcome = get_path_metric(
                    start,
                    neighbor,
                    battery_consumption_previous_stopover_node_table,
                    graph,
                )
                # if not get_path_metric(start, neighbor, battery_consumption_previous_stopover_node_table, graph)[0]:
                if not outcome:
                    battery_consumption_previous_stopover_node_table[neighbor] = None
                    battery_consumption_table[neighbor] = float("inf")
                    # visited_copy = visited.copy()
                    # test_neighbours(graph, end, battery_consumption_table, battery_consumption_previous_stopover_node_table, paths, visited, current_node, cost_so_far-weight)
                else:
                    heapq.heappush(priority_queue, (cost_so_far, neighbor))
            # here was outcome and all that
    return False


def get_path_metric(
    start: int,
    current_node: int,
    battery_consumption_previous_stopover_node_table: Dict[int, int],
    graph: nx.MultiGraph,
) -> bool:
    """A function that returns whether the path is feasible or not

    Args:
        start (int)
        current_node (int)
        battery_consumption_previous_stopover_node_table (Dict[int, int])
        graph (nx.MultiGraph)

    Returns:
        bool
    """
    path = construct_path(
        start, current_node, battery_consumption_previous_stopover_node_table
    )
    metric = 0
    for index in range(len(path) - 1):
        node1 = path[index]
        node2 = path[index + 1]
        step = get_edge_metric(graph, node1, node2)
        metric += step
        if node_is_cs(graph, node2):
            metric = 0
        if metric > 90:
            return False
    return True


def construct_path(
    start: int,
    end: int,
    battery_consumption_previous_stopover_node_table: Dict[int, int],
) -> List[int]:
    """a function that returns a path using the records of which node should preceed the inputted node when going along the most optimal route

    Args:
        start (int)
        end (int)
        battery_consumption_previous_stopover_node_table (Dict[int,int])

    Returns:
        List[int]
    """
    path = []
    current_node = end
    while current_node != start:
        path.append(current_node)
        current_node = battery_consumption_previous_stopover_node_table[current_node]
    path.append(start)
    path.reverse()
    return path
