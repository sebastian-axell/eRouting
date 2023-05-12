"""
BiS4EV
========


BiS4EV is a routing algorithm for electric vehicle. 

The code here is the implementation of BiS4EV for this project
"""


import networkx as nx
from path import Path
from auxilary import construct_edge_data, get_path_metric, node_is_cs
import heapq
from typing import Dict, List, Tuple, Union


def find_recursive_path_forward(
    graph: nx.MultiDiGraph, node: int, target: int
) -> List[int]:
    """A function that finds a path from the origin to the given node

    Args:
        graph (nx.MultiDiGraph)
        node (int)
        target (int)

    Returns:
        List[int]: path from origin to the given node
    """
    path = []
    while node != target:
        if isinstance(node, tuple):
            path.extend(list(node))
            node = node[-1]
        else:
            path.append(node)
        node = graph.nodes[node]["previous"]
    path.append(target)
    path.reverse()
    return path


def find_recursive_path_backward(
    graph: nx.MultiDiGraph, node: int, target: int
) -> List[int]:
    """A function that finds a path from the given node to the target

    Args:
        graph (nx.MultiDiGraph)
        node (int)
        target (int)

    Returns:
        List[int]: a path from the given node to the target
    """
    path = []
    while node != target:
        if isinstance(node, tuple):
            path.extend(list(node))
            node = node[-1]
        else:
            path.append(node)
        node = graph.nodes[node]["next"]
    if path and path[-1] != target:
        path.append(target)
    return path


def get_edges(edge_set: List[Tuple[int, int]]) -> List[int]:
    """A function that extracts all the edges from a set of node, node pairs constituting an edge

    Args:
        edge_set (List[Tuple[int, int]])

    Returns:
        List[int]
    """
    forward_nodes_edges = set()
    for element in edge_set:
        if len(element) > 2:
            for index in range(len(element) - 1):
                forward_nodes_edges.add((element[index], element[index + 1]))
        else:
            forward_nodes_edges.add(element)
    return forward_nodes_edges


def get_explored_graph(
    graph: nx.MultiGraph,
    forward_edges: List[Tuple[int, int]],
    backward_edges: List[Tuple[int, int]] = None,
) -> nx.MultiGraph:
    """A function that returns a nx.Multigraph based on explored edges

    Args:
        graph (nx.MultiGraph)
        forward_edges (List[Tuple[int,int]])
        backward_edges (List[Tuple[int,int]], optional). Defaults to None.

    Returns:
        nx.MultiGraph
    """
    forward_nodes_edges = get_edges(forward_edges)
    if backward_edges:
        backward_nodes_edges = get_edges(backward_edges)
        forward_nodes_edges.update(backward_nodes_edges)
    explored_graph = nx.MultiGraph(forward_nodes_edges)
    remove_list = []
    for edge in explored_graph.edges():
        neigbour1 = edge[0]
        neigbour2 = edge[1]
        try:
            edge_data = graph.get_edge_data(neigbour1, neigbour2)[0]
            explored_graph[neigbour1][neigbour2][0].update(edge_data)
        except:
            remove_list.append(edge)
    for edge in remove_list:
        explored_graph.remove_edge(edge[0], edge[1])
    explored_graph.update(graph.edges(data=True), graph.nodes(data=True))
    return explored_graph


def find_next_stop(
    graph: nx.MultiGraph,
    current_stop: int,
    search_path: List[int],
    end: int,
    charging_stops: List[int],
) -> bool:
    """Function to find the next optimal charging stop, recursively

    Args:
        graph (nx.MultiGraph)
        current_stop (int)
        search_path (List[int])
        end (int)
        charging_stops (List[int])

    Returns:
        bool: a bool to signify if the charging stops were found
    """
    # find which charging station works out of all charging stations
    possible_stations = extract_possible_stations(graph, current_stop, search_path)
    if possible_stations:
        # get the farthest station that is reachable (bc <= 90) and farthest (distance & reverse=True)
        next = sorted(
            possible_stations, key=lambda x: x[1].battery_consumption, reverse=True
        )[0]
        next_stop = next[1].end  # the end is the actual charging station
        next_stop_index = next[0]
        if get_path_metric(search_path[next_stop_index:], graph) <= 90:
            return True
        charging_stops.append(next_stop)
        return find_next_stop(
            graph, next_stop, search_path[next_stop_index:], end, charging_stops
        )
    else:
        return False


def extract_possible_stations(
    graph: nx.MultiGraph, current_stop: int, search_path: List[int]
) -> List[Tuple[int, Path]]:
    """A function to find all the reachable stations given the current stop and the remaining path

    Args:
        graph (nx.MultiGraph)
        current_stop (int)
        search_path (List[int])

    Returns:
        List[Tuple[int, Path]]
    """
    possible_stations = []
    charging_stations = [
        (index, node)
        for (index, node) in enumerate(search_path)
        if node_is_cs(graph, node)
    ]
    for index, charging_station in charging_stations:
        if charging_station == current_stop:
            continue
        path = search_path[: index + 1]
        bc = get_path_metric(path, graph)
        distance = get_path_metric(path, graph, "length")
        if bc <= 90:
            path = Path(current_stop, charging_station, path, distance)
            possible_stations.append((index, path))
    return possible_stations


def astar(start: int, goal: int, graph: nx.MultiGraph) -> List[int]:
    """An implementation of A*

    Args:
        start (int)
        goal (int)
        graph (nx.MultiGraph)

    Returns:
        List[int]
    """
    priority_queue = [(0, start)]
    predecessor = {}
    cost_from_start = {}
    predecessor[start] = None
    cost_from_start[start] = 0
    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)
        if current_node == goal:
            break
        for neighbour in graph.neighbors(current_node):
            cost_to_neighbour_via_current = (
                cost_from_start[current_node]
                + graph[current_node][neighbour][0]["battery_consumption"]
            )
            if (
                neighbour not in cost_from_start
                or cost_to_neighbour_via_current < cost_from_start[neighbour]
            ):
                cost_from_start[neighbour] = cost_to_neighbour_via_current
                priority = cost_to_neighbour_via_current + manhattan_dist(
                    goal, neighbour, graph
                )
                heapq.heappush(priority_queue, (priority, neighbour))
                predecessor[neighbour] = current_node
    return construct_path_astar(start, goal, predecessor)


def construct_path_astar(start: int, goal: int, predecessor: int) -> List[int]:
    """A function that constructs a path for A*

    Args:
        start (int)
        goal (int)
        predecessor (int)

    Returns:
        List[int]
    """
    path = []
    current_node = goal
    while current_node != start:
        path.append(current_node)
        current_node = predecessor[current_node]
    path.append(start)
    path.reverse()
    return path


def manhattan_dist(node1: int, node2: int, graph: nx.MultiGraph) -> int:
    """A function that returns the manhattan distance between two points

    Args:
        node1 (int)
        node2 (int)
        graph (nx.MultiGraph)

    Returns:
        int
    """
    x_coord_node1, y_coord_node1 = graph.nodes[node1]["x"], graph.nodes[node1]["y"]
    x_coord_node2, y_coord_node2 = graph.nodes[node2]["x"], graph.nodes[node2]["y"]
    return abs(x_coord_node1 - x_coord_node2) + abs(y_coord_node1 - y_coord_node2)


def _construct_edge_recur(
    datasets: List[Dict[str, Tuple[int, str, bool]]],
    current_data: Dict[str, Tuple[int, str, bool]],
) -> Dict[str, Tuple[int, str, bool]]:
    """A function that returns the cumulative data across a list of edges

    Args:
        datasets (List[Dict[str, Tuple[int,str, bool]]]): a list of edge datas
        current_data (Dict[str, Tuple[int,str, bool]]): the current accumulated version of the edge datas

    Returns:
        Dict[str, Tuple[int,str, bool]]: the accumulated version of the edge datas
    """
    if not datasets:
        return current_data
    current_data = construct_edge_data(datasets[0], current_data)
    return _construct_edge_recur(datasets[1:], current_data)


def _node_is_linking_node(node_data: Dict[str, Union[float, str, bool, int]]) -> bool:
    """A function to check if a node is a suitable linking node

    Args:
        node_data (Dict[str, Union[float, str, bool, int]]): node and its data

    Returns:
        bool: outcome of the check
    """
    if node_data["dis_f_cs"] + node_data["dis_b_cs"] < 90 or node_data["CS"]:
        return True
    return False


class bis_4_ev:
    def __init__(self, graph: nx.MultiGraph, start: int, target: int):
        """The construtor function of BiS4EV

        Args:
            graph (nx.MultiGraph): _description_
            start (int): _description_
            target (int): _description_
        """
        self.origin = start
        self.target = target
        self.name = "BiS4EV"
        self.graph = graph
        self.nodes = self.graph.nodes(data=True)
        self.forward_explored = set()
        self.backward_explored = set()
        self.forward_edges = set()
        self.backward_edges = set()
        self.path = None
        self.charging_stops = None

    def _extract_min(
        self, param: str, direction: str
    ) -> Tuple[int, Dict[str, Union[str, int, bool, float]]]:
        """a function that returns the next node to be examined for the specified direction

        Args:
            param (str)
            direction (str)

        Returns:
            Tuple[int, Dict[str, Union[str,int,bool, float]]]: returns a tuple of the node id and its data
        """
        sorted_list = sorted(self.nodes, key=lambda x: self.nodes[x[0]][param])
        index = 0
        candidate = None
        while True:
            candidate, data = sorted_list[index]
            if direction == "forward":
                if candidate in self.forward_explored:
                    index += 1
                    candidate = None
                    continue
            else:
                if candidate in self.backward_explored:
                    index += 1
                    candidate = None
                    continue
            if index == len(sorted_list):
                return -1, -1
            if candidate:
                return candidate, data

    def _pre_process_nodes(self):
        """A function that pre-processes the nodes ahead of path-finding in BiS4EV"""
        key_points = [self.origin, self.target]
        for node_id, node_data in self.nodes:
            node_data["dis_f"] = float("inf")
            node_data["dis_f_cs"] = float("inf")
            node_data["dis_b"] = float("inf")
            node_data["dis_b_cs"] = float("inf")
            node_data["previous"] = self.origin
            node_data["next"] = self.target
            if node_data["CS"] or node_id in key_points:
                node_data["previous_cs"] = node_id
            else:
                node_data["previous_cs"] = None
            if node_data["CS"] or node_id in key_points:
                node_data["next_cs"] = node_id
            else:
                node_data["next_cs"] = None
            node_data["is_stalled"] = False
        self.graph.nodes()[self.origin]["dis_f"] = 0
        self.graph.nodes()[self.origin]["dis_f_cs"] = 0
        self.graph.nodes()[self.target]["dis_b"] = 0
        self.graph.nodes()[self.target]["dis_b_cs"] = 0

    def find_route(self) -> Union[Tuple[int, int], Tuple[List[int], List[int]]]:
        """The function to find a path between the start and target

        Returns:
            Union[Tuple[int,int], Tuple[List[int], List[int]]]: returns the path and the charging stops along the path, or a flag to say no path was found
        """
        self._pre_process_nodes()
        # print(f"Going from {self.origin} to {self.target}")
        n = None  # meeting point
        back_trapped = False
        forward_trapped = False
        explored = set()
        u, u_data = self._extract_min("dis_f", "forward")
        v, v_data = self._extract_min("dis_b", "backward")
        # will an already testing link change
        while True:
            # print(len(explored))
            ### for the forward search
            if not forward_trapped:
                if not u_data["is_stalled"]:
                    if u_data["dis_f"] > 90 and u_data["dis_f_cs"] > 90:
                        # case a :: cannot be reached directly
                        # print("case a")
                        u_data["is_stalled"] = True
                        explored.add(u)
                    else:
                        # case b :: can be reached [all the way from the origin], i.e. u_data['dis_f'] <= 90 or u_data['dis_f_cs'] <= 90
                        # print("case b")
                        explored.add(u)
                        self._execute_case_b_forward(u, u_data)
                        if (
                            u == self.target or u in self.backward_explored
                        ):  # check the intermediate node
                            if self._is_valid_meeting_point():
                                n = u
                                break
                else:
                    explored.add(u)
                    self._set_node_param(u, "dis_f", float("inf"))
                    # we need to redirect the node via a charging point
                    # get the distance from u to each of these
                    # using edges we have trodded before
                    # charging_stations = [node for node in self.forward_explored if node_is_cs(self.graph, node)]
                    # forward_graph = get_explored_graph(self.graph, self.forward_edges)
                    # routes = []
                    # for station in charging_stations:
                    #     route = astar(u, station, forward_graph)
                    #     routes.append(route)
                    # cheapest = float('inf')
                    # for route in routes:
                    #     cost = self._get_path_metric(route)
                    #     if cost < cheapest:
                    #         smallest = route
                    #         cheapest = cost
                    # cost_to_station = cheapest
                    # if cost_to_station <= 90:
                    #         print("case c")
                    #         u_data['is_stalled'] = False
                    #         path_from_start = find_recursive_path_forward(self.graph, smallest[-1], self.origin)
                    #         cost_from_start = self._get_path_metric(path_from_start)
                    #         self._set_node_params(u,
                    #         (('previous', tuple(smallest[1:])[::-1]),
                    #         ('dis_f', cost_to_station + cost_from_start),
                    #         ('dis_f_cs', cost_to_station if not node_is_cs(self.graph, u) else 0),
                    #         ("previous_cs", smallest[-1] if not node_is_cs(self.graph, u) else u)
                    #         ))
                    #         # add edge from u to station
                    #         self._add_edge(smallest, cost_to_station)
                    # else:
                    #     # case d :: we cannot get to it at all
                    #     print("case d")
                    #     explored.add(u)
                    #     self._set_node_param(u, 'dis_f', float('inf'))

            ### backward search
            if not back_trapped:
                if not v_data["is_stalled"]:
                    if v_data["dis_b"] > 90 and v_data["dis_b_cs"] > 90:
                        # case a
                        # print("case a")
                        explored.add(v)
                        v_data["is_stalled"] = True
                    else:
                        # case b
                        # print("case b")
                        explored.add(v)
                        self._execute_case_b_backward(v, v_data)
                        if (
                            v == self.origin or v in self.forward_explored
                        ):  # check the intermediate node
                            if self._is_valid_meeting_point():
                                n = v
                                break
                else:
                    explored.add(v)
                    self._set_node_param(v, "dis_b", float("inf"))
                    # we need to redirect the node via a charging point
                    # charging_stations = [node for node in self.backward_explored if node_is_cs(self.graph, node)]
                    # backward_graph = get_explored_graph(self.graph, self.backward_edges)
                    # routes = []
                    # for station in charging_stations:
                    #     route = astar(v, station, backward_graph)
                    #     routes.append(route)
                    # cheapest = float('inf')
                    # for route in routes:
                    #     cost = self._get_path_metric(route)
                    #     if cost < cheapest:
                    #         smallest = route
                    #         cheapest = cost
                    # cost_to_station = cheapest
                    # if cost_to_station <= 90:
                    #     print("case c back")
                    #     v_data['is_stalled'] = False
                    #     path_from_station_to_end = find_recursive_path_backward(self.graph, smallest[-1], self.target)
                    #     cost_from_station_to_end = self._get_path_metric(path_from_station_to_end)
                    #     self._set_node_params(v,
                    #             (('next', tuple(smallest[1:])), # this is in => direction
                    #             ('dis_b', cost_to_station + cost_from_station_to_end),
                    #             ('dis_b_cs', cost_to_station if not node_is_cs(self.graph, v) else 0),
                    #             ("next_cs", smallest[-1] if not node_is_cs(self.graph, v) else v)
                    #             ))
                    #     self._add_edge(smallest,cost_to_station)
                    #     # add edge from station to
                    # else:
                    #     # case d :: we cannot get to it at all
                    #     print("case d forward")
                    #     explored.add(v)
                    #     self._set_node_param(v, 'dis_b', float('inf'))
            u, u_data = self._extract_min("dis_f", "forward")
            v, v_data = self._extract_min("dis_b", "backward")
            if (forward_trapped and back_trapped) or len(explored) == len(self.nodes):
                break
            if v_data["dis_b"] == float("inf"):
                back_trapped = True
            else:
                back_trapped = False
            if u_data["dis_f"] == float("inf"):
                forward_trapped = True
            else:
                forward_trapped = False
        if n:
            print("Path found")
            return self.path, self.charging_stops
        else:
            print("Path not found")
            return -1, -1

    # used for the second implementation version of case c
    # def _add_edges(self, smallest, type):
    #     for index in range(len(smallest)-1):
    #         node1 = smallest[index]
    #         node2 = smallest[index+1]
    #         if type == "backward":
    #             self.backward_edges.add((node1, node2))
    #         elif type == "forward":
    #             self.forward_edges.add((node1, node2))

    def _add_edge(self, smallest: List[int], cost: int):
        """function that adds edges given a path, smallest

        Args:
            smallest (List[int]): a path from which edges should be added
            cost (int): cost of traversing the edge
        """
        length = self._get_path_metric(smallest, "length")
        self.graph.add_edge(smallest[0], smallest[-1])
        edge_data = {
            "battery_consumption": cost,
            "highway": "road",
            "length": length,
            "lit": "no",
            "tunnel": "no",
        }
        self.graph[smallest[0]][smallest[-1]][0].update(edge_data)

    def _execute_case_b_backward(
        self, v: int, v_data: Dict[str, Union[str, int, float]]
    ):
        """A function that executes case b in the backward search direction

        Args:
            v (int)
            v_data (Dict[str, Union[str,int, float]])
        """
        self.backward_explored.add(v)
        # started editing things now
        # when we add v, the end of the path will already be there and with v there as well the edge we added will become accessible, even in a subgraph
        # the previous step can either be a collection of nodes
        if isinstance(v_data["next"], tuple):
            self.backward_edges.add((v,) + v_data["next"])
        else:
            # or a single node
            if not v == v_data["next"]:
                self.backward_edges.add((v, v_data["next"]))
        # update distances of neighbours of v when going via v
        for edge in self.graph.edges(v):
            self._relax_edges_back(edge)

    def _is_valid_meeting_point(self) -> bool:
        """A function that checks if there is a valid linking node

        Returns:
            bool: the outcome of the search
        """
        path = self._get_shortest_feasible_path()
        if path == -1:
            return False
        else:
            self.path = path
            self.charging_stops = self._apply_charging_policy()
            return True

    def _execute_case_b_forward(
        self, u: int, u_data: Dict[str, Union[str, int, float]]
    ):
        """A function that executes case b in the forward search direction

        Args:
            u (int)
            u_data (Dict[str, Union[str,int, float]])
        """
        self.forward_explored.add(u)
        if isinstance(u_data["previous"], tuple):
            self.forward_edges.add(u_data["previous"] + (u,))
        else:
            if not u_data["previous"] == u:  # to avoid saving (3611, 3611)
                self.forward_edges.add((u_data["previous"], u))
        # update distances of neighbours of u when going via u
        for edge in self.graph.edges(u):
            self._relax_edges(edge)

    def _apply_charging_policy(self) -> List[int]:
        """Function that applies the charging policy to the path

        Returns:
            List[int]: charging stops along the path
        """
        is_charging_needed = self.check_path_consumption()
        charging_stops_along_path = []
        if not is_charging_needed:
            self.charging_time = 0
        else:
            charging_stops_along_path.append(self.path[0])
            find_next_stop(
                self.graph,
                self.path[0],
                self.path,
                self.path[-1],
                charging_stops_along_path,
            )
        return charging_stops_along_path

    def check_path_consumption(self) -> bool:
        """A function that checks if charging is needed when traversing the found path

        Returns:
            bool: flag indicating whether charge is needed
        """
        bc = 0
        charging_needed = False
        for index in range(len(self.path) - 1):
            bc += self.graph.get_edge_data(self.path[index], self.path[index + 1])[0][
                "battery_consumption"
            ]
            if bc >= 90:
                charging_needed = True
                break
        return charging_needed

    def _get_shortest_feasible_path(self):
        """Function that checks if there are any suitable linking nodes that have been explored. If so the path going through it is returned; it not a flag of -1 is
        returned

        Returns: path or flag
        """
        estimate = float("inf")
        path = None
        linking_node = None
        for node in self.forward_explored & self.backward_explored:  # node both in sets
            node_data = self.nodes[node]
            distance_f = node_data["dis_f"]
            distance_b = node_data["dis_b"]
            # find a linking node and make a path
            # estimate keeps track of the the shortest thus far
            if _node_is_linking_node(node_data) and estimate > distance_b + distance_f:
                estimate = distance_f + distance_b
                linking_node = node
                path = self._construct_path(node)
        if (
            not linking_node or path == -1
        ):  # path[0] != self.origin or path[-1] != self.target:
            return -1
        return path

    def _construct_path(self, node: int) -> List[int]:
        """Function that constructs a path from the origin to the given node, and from the given node to the target

        Args:
            node (int): the suitable linking node

        Returns:
            List[int]: the path from the origin to the target going via node
        """
        path = find_recursive_path_forward(self.graph, node, self.origin)
        path_end = find_recursive_path_backward(self.graph, node, self.target)
        path.extend(path_end[1:])
        return path

    def _get_path_metric(
        self, path: list[int], variable="battery_consumption", breakpoint=False
    ) -> float:
        """Function that returns a given metric of a given path

        Args:
            path (list[int]): path
            variable (str, optional): . Defaults to "battery_consumption".
            breakpoint (bool, optional): . Defaults to False.

        Returns:
            float : the metric of the path
        """
        metric = 0
        for index in range(len(path) - 1):
            node1 = path[index]
            node2 = path[index + 1]
            metric += self._get_edge_data(node1, node2, variable)
            if breakpoint and metric >= 90:
                return float("inf")
        return metric

    def _get_edge_data(
        self, u: int, v: int, data="battery_consumption", key=0
    ) -> Union[str, float, bool, int, None]:
        """Function that returns a given dataset of an edge in the graph

        Args:
            u (int): the first node
            v (int): the second node
            data (str, optional): what data should be returned. Defaults to 'battery_consumption'.
            key (int, optional): what edge key should be used. Defaults to 0.

        Returns:
            Union[str, float, bool, int, None]: the specified edge data or none
        """
        try:
            return self.graph.get_edge_data(u, v)[key][data]
        except KeyError:
            key = list(self.graph.get_edge_data(u, v).keys())[0]
        return self.graph.get_edge_data(u, v)[key][data]

    def _set_node_params(
        self, node_id: int, params: List[Tuple[str, Union[str, int, bool, float]]]
    ):
        """a function that sets the given parameters of a given node

        Args:
            node_id (int)
            params (List[Tuple[str, Union[str, int, bool, float]]])
        """
        node = self.nodes[node_id]
        for key, value in params:
            node[key] = value

    def _set_node_param(
        self, node_id: int, key: str, value: Union[str, bool, int, float]
    ):
        """function that sets a given parameter value of a given node

        Args:
            node_id (int): _description_
            key (str): _description_
            value (Union[str,bool,int]): _description_
        """
        node = self.nodes[node_id]
        node[key] = value

    def _get_node_param(
        self, node: int, param: Union[str, int, bool, float]
    ) -> Union[str, bool, int, float]:
        """a function that returns a given parameter of a node's data

        Args:
            node (int)
            param (Union[str, int, bool, float])

        Returns:
            Union[str, int, bool, float]
        """
        return self.nodes[node][param]

    def _relax_edges(self, edge: Tuple[int, int]):
        """a function that relaxes a given edge in the forward search direction

        Args:
            edge (Tuple[int,int])
        """
        u, x = edge
        # dis is alwats better energy consumption from s to u
        walk_from_u_to_x = self.nodes[u]["dis_f"] + self._get_edge_data(u, x)
        walk_from_u_cs_to_x = self.nodes[u]["dis_f_cs"] + self._get_edge_data(u, x)
        if self.nodes[x]["dis_f"] > walk_from_u_to_x:
            # updating the distance from the origin to x by settings its distance the distance via u
            self._set_node_params(x, (("dis_f", walk_from_u_to_x), ("previous", u)))
            # check if it is a CS
            if self._get_node_param(x, "CS"):
                self._set_node_params(x, (("previous_cs", x), ("dis_f_cs", 0)))
            else:
                self._set_node_params(
                    x,
                    (
                        ("previous_cs", self._get_node_param(u, "previous_cs")),
                        ("dis_f_cs", walk_from_u_cs_to_x),
                    ),
                )

    def _relax_edges_back(self, edge: Tuple[int, int]):
        """a function that relaxes a given edge in the forward search direction

        Args:
            edge (Tuple[int, int])
        """
        # dis is always better energy consumption from s to u
        v, x = edge
        walk_from_v_to_x = self.nodes[v]["dis_b"] + self._get_edge_data(v, x)
        walk_from_v_cs_to_x = self.nodes[v]["dis_b_cs"] + self._get_edge_data(v, x)
        if self.nodes[x]["dis_b"] > walk_from_v_to_x:
            self._set_node_params(x, (("dis_b", walk_from_v_to_x), ("next", v)))
            # check if it is a CS
            if self._get_node_param(x, "CS"):
                self._set_node_params(x, (("next_cs", x), ("dis_b_cs", 0)))
            else:
                self._set_node_params(
                    x,
                    (
                        ("next_cs", self._get_node_param(v, "next_cs")),
                        ("dis_b_cs", walk_from_v_cs_to_x),
                    ),
                )
