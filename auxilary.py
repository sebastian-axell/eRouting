from pyrosm import get_data, OSM
from random import choice
from graph import Graph, GraphException
import networkx as nx
import osmnx as ox
import re
import os
from typing import Dict, List, Set, Union


def read_paths(function_type: str, route_type: str) -> List[List[int]]:
    """A function that read paths from paths sub-directory in the current directory

    Args:
        function_type (str)
        route_type (str)

    Returns:
        List[List[int]]
    """
    file_names = f"^{function_type}_.*_{route_type}.txt$"
    destination = f"./paths/{function_type}/"
    regex = re.compile(file_names)
    paths = []
    for root, dirs, files in os.walk(destination):
        for file in files:
            if regex.match(file):
                load_path(destination, paths, file)
    return paths


def load_path(destination: str, paths: List[List[int]], file: str):
    """A function that loads a path

    Args:
        destination (str)
        paths (List[List[int]])
        file (str)
    """
    with open(destination + file) as f:
        # path = dict()
        path_list = []
        for line in f.readlines():
            path_list.append(int(line.strip()))
        # path[file] = path_list
        paths.append(path_list)
        f.close()


def construct_contraction_graph(graph: nx.MultiGraph) -> nx.MultiGraph:
    """A function that contracts a pre-processed graph

    Args:
        graph (nx.MultiGraph)

    Raises:
        GraphException: raises a graphException if the graph has not been pre-processed

    Returns:
        nx.MultiGraph
    """
    print("Preparing graph...")
    if graph.name != "processed":
        raise GraphException("Graph has to be pre-processed first")
    contracted_graph = graph.copy()
    order_nodes(contracted_graph)
    contract_nodes(contracted_graph)
    contracted_graph.name = "contracted"
    return contracted_graph


def expand_path(
    contracted_graph: nx.MultiGraph, original_graph: nx.MultiGraph, path: List[int]
) -> List[int]:
    """a function that re-adds contracted nodes to a path produced by BiS4EV. Assumes the graph has been expanded

    Args:
        contracted_graph (nx.MultiGraph)
        original_graph (nx.MultiGraph)
        path (List[int])

    Raises:
        GraphException: if the graph was not contracted, the path cannot be expanded
        ValueError

    Returns:
        List[int]
    """
    if contracted_graph.name != "contracted":
        raise GraphException(
            "Graph has to be contracted in order for path to be expanded"
        )
    full_path = path.copy()
    added = 0
    add_list = []
    for index in range(len(path) - 1):
        node1 = path[index]
        node2 = path[index + 1]
        try:
            if original_graph.get_edge_data(node1, node2) is None:
                raise ValueError("Cmon")
        except ValueError:
            contracted_node = contracted_graph.nodes[node1]["contracted_from"]
            add_list.append((index + 1, contracted_node))
    for pair in add_list:
        add_index = pair[0]
        add_node = pair[1]
        full_path.insert(add_index + added, add_node)
        added += 1
    return full_path


def contract_nodes(graph: nx.MultiGraph):
    """contracts nodes in the given graph

    Args:
        graph (nx.MultiGraph)
    """
    nodes = graph.nodes(data=True)
    contraction_nodes = []
    for node_id, node_data in nodes:
        if node_can_be_contracted(node_data) and no_contracted_neighbours(
            graph, node_id
        ):
            contraction_nodes.append(node_id)
            update_neighbouring_nodes(graph, node_id)
    for node in contraction_nodes:
        contract_node(graph, node)


def contract_node(graph: nx.MultiGraph, node_id: str):
    """contracts a given node

    Args:
        graph (nx.MultiGraph)
        node_id (str)
    """
    neighbours = list(graph.neighbors(node_id))
    # for each pair of neighbours
    for index in range(len(neighbours) - 1):
        first_neighbour = neighbours[index]  # this is a node id
        second_neighbour = neighbours[index + 1]  # this is a node id
        # print("contracting", node_id, first_neighbour, second_neighbour)
        add_edge(graph, first_neighbour, second_neighbour, node_id)
    if graph.degree(node_id) == 3:
        first_neighbour = neighbours[0]  # this is a node id
        second_neighbour = neighbours[-1]  # this is a node id
        # print("contracting", node_id, first_neighbour, second_neighbour)
        add_edge(graph, first_neighbour, second_neighbour, node_id)
    # Remove node_2b_del from the graph
    graph.remove_node(node_id)


def update_neighbouring_nodes(graph: nx.MultiGraph, node_2b_del: int):
    """updates the neighbours of the given node; it is set to be contracted

    Args:
        graph (nx.MultiGraph)
        node_2b_del (int)
    """
    for neighbour in graph.neighbors(node_2b_del):
        # to say that node_id will be removed
        # => don't contract its neighbours!
        graph.nodes[neighbour][
            "contracted_from"
        ] = node_2b_del  # to find contracted nodes
        # node_2b_del used to be a neighbour (it will be removed)
        graph.nodes[neighbour]["contraction_count"] += 1
    graph.nodes[node_2b_del]["contraction_count"] += 1


def no_contracted_neighbours(graph: nx.MultiGraph, node: int) -> bool:
    """checks if the node has any contracted neighbours

    Args:
        graph (nx.MultiGraph)
        node (int)

    Returns:
        bool
    """
    for neighbour in graph.neighbors(node):
        # get the node data
        node = graph.nodes[neighbour]
        # it is a neighbour of a node that will be contracted
        if node["contraction_count"] != 0:
            # it cannot be contracted
            return False
    return True


def node_can_be_contracted(node_data: Dict[str, Union[str, bool, int, float]]) -> bool:
    """a check to see if a node can be contracted

    Args:
        node_data (Dict[str, Union[str,bool,int]])

    Returns:
        bool
    """
    # node can be contracted if it does is complaint with:
    # 1 :: No CSs
    # 2 :: No nodes with degree(n)>= 4
    # 3 :: No nodes with neighbour has already been contraced
    # -> not contracting the nodes whose neighbor nodes have been contracted
    # 4 :: if the battery consumption between the nodes it connected is greater than 90
    if (
        not node_data["CS"]
        and 1 < node_data["neigbour_count"] < 4
        and node_data["dist_ok"]
    ):
        return True
    return False


def add_edge(graph: nx.MultiGraph, neigbour1: int, neigbour2: int, node_2b_del: int):
    """adds an edge between the two neighbours of a node set to be removed

    Args:
        graph (nx.MultiGraph)
        neigbour1 (int)
        neigbour2 (int)
        node_2b_del (int)
    """
    first_edge_data = get_edge_info(graph, node_2b_del, neigbour1)
    second_edge_data = get_edge_info(graph, node_2b_del, neigbour2)
    new_edge_data = construct_edge_data(first_edge_data, second_edge_data)
    # Move create the edge between neighbour1 and neigbour2
    graph.add_edge(neigbour1, neigbour2)
    graph[neigbour1][neigbour2][0].update(new_edge_data)


def order_nodes(graph: nx.MultiGraph):
    """order the nodes in ascending order of importance

    Args:
        graph (nx.MultiGraph)
    """
    # sort nodes
    nodes = graph.nodes(data=True)
    sorted_nodes = sorted(nodes, key=lambda x: nodes[x[0]]["importance_degree"])
    # prepare sorted_graph
    sorted_graph = graph.copy()
    # clear its nodes
    sorted_graph.remove_nodes_from(list(graph.nodes()))
    # create temp graph
    temp = nx.Graph()
    temp.add_nodes_from(sorted_nodes)
    # add sorted nodes
    sorted_graph.nodes = temp.nodes
    # add its edges
    sorted_graph.add_edges_from(graph.edges(data=True))
    graph = sorted_graph


def construct_graph(
    location: str,
    times: int = 1,
    seed: int = 2020,
    additional_charging_stations: List[int] = None,
) -> Graph:
    """contructs a network graph of the given location using the given conditions

    Args:
        location (str)
        times (int, optional). Defaults to 1.
        seed (int, optional). Defaults to 2020.
        additional_charging_stations (List[int], optional). Defaults to None.

    Returns:
        Graph
    """
    region = get_data(location)
    print("Converting OSM data into graph...")
    osm = OSM(region)
    nodes, edges = osm.get_network(nodes=True, network_type="driving")
    graph = osm.to_graph(nodes, edges, graph_type="networkx")
    graph = graph.to_undirected()  # we will be assuming bidirectional roads
    charging_stations = get_charging_stations(additional_charging_stations, osm, graph)
    graph = Graph(graph, charging_stations, times, seed)
    print("Graph completed")
    return graph


def get_charging_stations(
    additional_charging_stations: List[int], osm: OSM, graph: nx.MultiGraph
) -> Set[int]:
    """gets the node IDs of all the nodes set to be charging stations

    Args:
        additional_charging_stations (List[int])
        osm (OSM)
        graph (nx.MultiGraph)

    Returns:
        Set[int]
    """
    charging_stations = osm.get_pois(custom_filter={"amenity": ["charging_station"]})
    cs_stations = charging_stations[["lon", "lat"]].dropna()
    node_approxs = []
    for cs_index in range(cs_stations.shape[0]):
        station = cs_stations.iloc[cs_index]
        node_approxs.append(ox.distance.nearest_nodes(graph, station[0], station[1]))
    if additional_charging_stations:
        node_approxs.extend(additional_charging_stations)
    cleansed_list = set(node_approxs)
    return cleansed_list


def get_edge_info(
    graph: nx.MultiGraph, u: int, v: int
) -> Dict[str, Union[str, bool, int, float]]:
    """returns the edge data of a given edge

    Args:
        graph (nx.MultiGraph)
        u (int)
        v (int)

    Returns:
        Dict[str, Union[str,bool,int]]
    """
    edge_info = ["length", "battery_consumption", "lit", "tunnel", "highway"]
    # super weak assumption that they will always be 0 (key)
    edge_data = graph.get_edge_data(u, v)[0]
    # edge_keys = edge_data.keys()
    edge_data = {key: edge_data.get(key) for key in edge_info}
    # edge_data = {key: edge_data.get(key) for key in edge_keys}
    for key in ["lit", "tunnel"]:
        if edge_data[key] == "yes":
            edge_data[key] = True
        else:
            edge_data[key] = False
    return edge_data


def construct_edge_data(
    data1: Dict[str, Union[str, bool, int, float]],
    data2: Dict[str, Union[str, bool, int, float]],
) -> Dict[str, Union[str, bool, int, float]]:
    """combines two edge's datas

    Args:
        data1 (Dict[str, Union[str,bool,int]])
        data2 (Dict[str, Union[str,bool,int]])

    Returns:
        Dict[str, Union[str,bool,int]]
    """
    numbers = ["length", "battery_consumption"]
    bools = ["lit", "tunnel"]
    new_edge = {key: data1.get(key) + data2.get(key) for key in numbers}
    for key in bools:
        new_edge[key] = data1[key] & data2[key]
    new_edge["key"] = 0
    return new_edge

graph_settings = {
    "show": False,
    "close": False,
    "bgcolor": "#FFFFFF",  # background color of the plot
    "node_color": "#FF7F50",  # color of the nodes
    "node_size": 1,  # size of the nodes: if 0, skip plotting them
    "node_alpha": None,  # opacity of the
    "node_edgecolor": "none",  # color of the nodes' markers' borders
    "node_zorder": 1,  # zorder to plot nodes: edges are always 1
    "edge_color": "#32CD32",  # color of the edges)
    "edge_linewidth": 1,
    "figsize": (18,18)
}

def print_graph(graph):
    fig,ax = ox.plot_graph(
        graph, show=graph_settings['show'], 
        close=graph_settings['close'], 
        bgcolor=graph_settings['bgcolor'],  # background color of the plot
        node_color=graph_settings['node_color'],  # color of the nodes
        node_size=graph_settings['node_size'],  # size of the nodes: if 0, skip plotting them
        node_alpha=graph_settings['node_alpha'],  # opacity of the
        node_edgecolor=graph_settings['node_edgecolor'],  # color of the nodes' markers' borders
        node_zorder=graph_settings['node_zorder'],  # zorder to plot nodes: edges are always 1
        edge_color=graph_settings['edge_color'],  # color of the edges)
        edge_linewidth=graph_settings['edge_linewidth'],
        figsize=graph_settings['figsize']
    )
    return fig, ax

def print_route(graph: nx.MultiGraph, route: List[int], name: str):
    """prints a route to the test directory of the data director of the current directory to a given name

    Args:
        graph (nx.MultiGraph)
        route (List[int])
        name (str)
    """
    print_graph(graph, "routes", route, 'r')[0].savefig("./data/test/" + name + ".png")

def calculate_charging_time_dijkstra(charging_stops: List[int]) -> float:
    """returns the total charging time

    Args:
        charging_stops (List[int])

    Returns:
        float
    """
    total = 0
    for node, amount in charging_stops:
        total += charging_time(round(amount), 100)
    return total


def calculate_charging_time_bis4ev(
    graph: nx.MultiGraph, path: List[int], charging_stops_along_path: List[int]
) -> float:
    """calculates the total charging time to BiS4EV

    Args:
        graph (nx.MultiGraph)
        path (List[int])
        charging_stops_along_path (List[int])

    Returns:
        float
    """
    charge_time = 0
    sub_graph = graph.subgraph(path)
    for index in range(len(charging_stops_along_path) - 1):
        pair = (charging_stops_along_path[index], charging_stops_along_path[index + 1])
        path = nx.shortest_path(
            sub_graph, source=pair[0], target=pair[1], weight="battery_consumption"
        )
        bc = get_path_metric(path, graph)
        state_of_charge = 100 - bc
        if bc <= 70:
            # charge to 80 %
            charge_time += charging_time(round(state_of_charge), 80)
        elif 70 < bc <= 90:
            # charge to 100 %
            charge_time += charging_time(round(state_of_charge), 100)
    return charge_time


def get_edge_metric(
    graph: nx.MultiGraph, node1: int, node2: int, metric: str = "battery_consumption"
) -> Dict[str, Union[int, bool, str, float]]:
    """returns a given edge metric for a given edge

    Args:
        graph (nx.MultiGraph)
        node1 (int)
        node2 (int)
        metric (str, optional). Defaults to 'battery_consumption'.

    Returns:
        Dict[str, Union[int, bool, str, float]]
    """
    return graph.get_edge_data(node1, node2)[0][metric]


def node_is_cs(graph: nx.MultiGraph, node: int) -> bool:
    """returns a check to see if a node is charging stationn

    Args:
        graph (nx.MultiGraph)
        node (int)

    Returns:
        bool
    """
    return graph.nodes()[node]["CS"]


def get_path_metric(
    path: List[int], graph: nx.MultiGraph, variable: str = "battery_consumption"
) -> float:
    """returns a numeric metric for a given path

    Args:
        path (List[int])
        graph (nx.MultiGraph)
        variable (str, floatoptional). Defaults to 'battery_consumption'.

    Returns:
        float
    """
    metric = 0
    for index in range(len(path) - 1):
        node1 = path[index]
        node2 = path[index + 1]
        # metric += graph.get_edge_data(node1, node2)[0][variable]
        metric += get_edge_metric(graph, node1, node2, variable)
    return metric


def calculate_travel_time(graph: nx.MultiGraph, path: List[int]) -> float:
    """Calculates the total travel time for traversing a path

    Args:
        graph (nx.MultiGraph)
        path (List[int])

    Returns:
        float
    """
    sub_graph = graph.subgraph(path)  # get a subgraph
    sub_graph = ox.speed.add_edge_speeds(sub_graph)  # adds speed_kph edge attribute
    sub_graph = ox.speed.add_edge_travel_times(
        sub_graph
    )  # adds travel_time edge attribute (in seconds)
    travel_time = get_path_metric(path, graph, "travel_time")
    return travel_time


def charging_time(
    state_of_charge: int, charge_limit: int, time: float = 0.0, type: int = 1
) -> float:
    """calculates the charging time for a given SoC to a given point

    Assumes a 150kWh charge for type 1 (DC Fast Charge test)
    Function to return charging times for vehicles based on what the limit is

    0 -> 30 % :: 12 mins => [0,30] it takes on average 0.4 min/units
    30 -> 40 % :: 4 mins => [30,40] it takes on average 0.4 min/units
    40 -> 50 % :: 4 mins => [40,50] it takes on average 0.4 min/units
    50 -> 80 % :: 20 mins => [50,80] it takes on average 0.666 min/units
    80 -> 100 % :: 25 mins => [80,100] it takes on average 1.25 min/units

    Args:
        state_of_charge (int)
        charge_limit (int)
        time (float, optional). Defaults to 0.
        type (int, optional). Defaults to 1.

    Returns:
        float
    """
    if state_of_charge == charge_limit:
        return time
    if type == 1:
        if state_of_charge in range(0, 30):
            time += (30 - state_of_charge) * 0.4 * 60
            state_of_charge = 30
        elif state_of_charge in range(30, 40):
            time += (40 - state_of_charge) * 0.4 * 60
            state_of_charge = 40
        elif state_of_charge in range(40, 50):
            time += (50 - state_of_charge) * 0.4 * 60
            state_of_charge = 50
        elif state_of_charge in range(50, 80):
            time += (80 - state_of_charge) * 0.67 * 60
            state_of_charge = 80
        elif state_of_charge in range(80, 100):
            time += (100 - state_of_charge) * 1.25 * 60
            state_of_charge = 100
        else:
            print("not in any...", state_of_charge)
            return
        return charging_time(state_of_charge, charge_limit, time, type)
    else:
        pass


def get_colours(size: int) -> List[str]:
    """returns a list of randomly generated hex code colours

    Args:
        size (int)

    Returns:
        List[str]
    """
    return [
        "#" + "".join([choice("0123456789ABCDEF") for j in range(6)])
        for i in range(size)
    ]

def print_graph(graph, type="default", routes=None, rc=None):
    # colour scheme
        # background : white
        # points of interest: Navy blue (#001f3f)
        # nodes: Coral (#FF7F50)
        # edge_color: Lime green (#32CD32)
    graph_settings = {
        "show": False,
        "close": False,
        "bgcolor": "#FFFFFF",  # background color of the plot
        "node_color": "#FF7F50",  # color of the nodes
        "node_size": 1,  # size of the nodes: if 0, skip plotting them
        "node_alpha": None,  # opacity of the
        "node_edgecolor": "none",  # color of the nodes' markers' borders
        "node_zorder": 1,  # zorder to plot nodes: edges are always 1
        "edge_color": "#32CD32",  # color of the edges)
        "edge_linewidth": 1,
        "figsize": (18,18)
    }

    if type == "default":  
        fig,ax = ox.plot_graph(
            graph, 
            show=graph_settings['show'], 
            close=graph_settings['close'], 
            bgcolor=graph_settings['bgcolor'],  # background color of the plot
            node_color=graph_settings['node_color'],  # color of the nodes
            node_size=graph_settings['node_size'],  # size of the nodes: if 0, skip plotting them
            node_alpha=graph_settings['node_alpha'],  # opacity of the
            node_edgecolor=graph_settings['node_edgecolor'],  # color of the nodes' markers' borders
            node_zorder=graph_settings['node_zorder'],  # zorder to plot nodes: edges are always 1
            edge_color=graph_settings['edge_color'],  # color of the edges)
            edge_linewidth=graph_settings['edge_linewidth'],
            figsize=graph_settings['figsize']
        )
        return fig, ax
    elif type == "routes":   
        fig,ax = ox.plot_graph_routes(
            graph, routes=routes, route_colors=rc, route_linewidth=6, 
            show=graph_settings['show'], 
            close=graph_settings['close'], 
            bgcolor=graph_settings['bgcolor'],  # background color of the plot
            node_color=graph_settings['node_color'],  # color of the nodes
            node_size=graph_settings['node_size'],  # size of the nodes: if 0, skip plotting them
            node_alpha=graph_settings['node_alpha'],  # opacity of the
            node_edgecolor=graph_settings['node_edgecolor'],  # color of the nodes' markers' borders
            node_zorder=graph_settings['node_zorder'],  # zorder to plot nodes: edges are always 1
            edge_color=graph_settings['edge_color'],  # color of the edges)
            edge_linewidth=graph_settings['edge_linewidth'],
            figsize=graph_settings['figsize']
        )
        return fig, ax
    elif type == "route":
        fig,ax = ox.plot_graph_route(
        graph, route=routes, route_color='purple', route_linewidth=6, 
        show=graph_settings['show'], 
        close=graph_settings['close'], 
        bgcolor=graph_settings['bgcolor'],  # background color of the plot
        node_color=graph_settings['node_color'],  # color of the nodes
        node_size=graph_settings['node_size'],  # size of the nodes: if 0, skip plotting them
        node_alpha=graph_settings['node_alpha'],  # opacity of the
        node_edgecolor=graph_settings['node_edgecolor'],  # color of the nodes' markers' borders
        node_zorder=graph_settings['node_zorder'],  # zorder to plot nodes: edges are always 1
        edge_color=graph_settings['edge_color'],  # color of the edges)
        edge_linewidth=graph_settings['edge_linewidth'],
        figsize=graph_settings['figsize']
    )
    return fig, ax