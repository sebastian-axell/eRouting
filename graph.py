"""
Graph
========

Graph class developed for this project to simplify working with a networkx graph.

Handles pre-processing and adding labels to the nodes of the graph as needed

"""

from typing import Dict, List, Tuple, Union
import networkx as nx
import pandas as pd
from math import sqrt
from random import choice, seed
from ListDict import ListDict


def get_consumption_percentage(distance: int) -> float:
    """a function to get the consumption percentage from traversing a given distance.
    Based on the Volkswagen ID.4 model.

    Args:
        distance (int)

    Returns:
        float
    """
    efficiency = 1.87654  # kWh per 100 km
    battery_capcity = 77.0  # kWh
    energy_consumed = (distance / 100) * efficiency
    percentage_used = energy_consumed / battery_capcity
    return percentage_used


def calculate_importance_degree(node: int) -> float:
    """returns the importance degree of a given node

    Args:
        node (int)

    Returns:
        float
    """
    calc_dict = {
        "contraction_count": 1 / 3,
        "neigbour_count": 1 / 3,
        "edge_difference": 1 / 3,
    }
    importance = 0
    for key in calc_dict:
        importance += node[key] * calc_dict[key]
    return importance


class GraphException(Exception):
    """A class used to represent an issue in the graph."""

    pass


class Graph:
    def __init__(
        self,
        graph: nx.MultiGraph,
        charging_stations: List[int],
        times: int,
        seed_nbr: int,
    ):
        """constructor method for the Graph class

        Args:
            graph (nx.MultiGraph)
            charging_stations (List[int])
            times (int)
            seed_nbr (int)
        """
        self.graph = graph
        self.charging_stations = charging_stations
        self.data = pd.DataFrame(self.graph.nodes(data=True))
        self.times = times
        seed(seed_nbr)

    def get_pre_processed_graph(self) -> nx.MultiGraph:
        """returns a pre-processed version of the graph

        Returns:
            nx.MultiGraph
        """
        print("Processing graph")
        self._pre_process_nodes()
        return self.graph

    def get_nodes(self, data=False) -> List[int]:
        """returns the nodes of the graph

        Args:
            data (bool, optional). Defaults to False.

        Returns:
            List[int]
        """
        nodes = self.graph.nodes(data=data)
        return nodes

    def get_node(self, node_id: int) -> int:
        """returns the data of a given node

        Args:
            node_id (int)

        Returns:
            int
        """
        node = self.graph.nodes[node_id]
        return node

    def get_edges(self) -> List[Tuple[int, int]]:
        """returns the graph's edges

        Returns:
            List[Tuple[int,int]]
        """
        edges = self.graph.edges(data=True)
        return edges

    def get_coords(self) -> pd.DataFrame:
        """returns a pandas dataframe of the graph's nodes' long and lat coordinates

        Returns:
            pd.DataFrame
        """
        ##lat = Y lon = X
        tags = ["x", "y"]  # y is lat, x is lon
        coords = [
            {"lat" if tag == "y" else "lon": dict(point)[tag] for tag in tags}
            for point in self.data[1]
        ]
        return pd.DataFrame(coords)

    def _pre_process_nodes(self):
        """function that carries out all of the pre-processing"""
        self._add_battery_consumption()
        self._add_charging_stations()
        for node_id in self.get_nodes():
            self._add_node_labels(node_id)
        self.graph.name = "processed"

    def _add_node_labels(self, node_id: int):
        """function that adds node labels to a given node

        Args:
            node_id (int)
        """
        node = self.get_node(node_id)
        neighbours = self._get_neighbouring_nodes(
            node_id
        )  # list of nodes it is connected to
        neighbours_count = len(neighbours)
        node["contraction_count"] = 0
        node["charging_stop"] = False
        node["neigbour_count"] = neighbours_count
        if neighbours_count == 2:
            node["edge_difference"] = -1
            node["dist_ok"] = self._check_connection_cost(node_id, neighbours)
        elif neighbours_count == 3:
            node["edge_difference"] = 0
            node["dist_ok"] = self._check_connection_cost(node_id, neighbours)
        elif neighbours_count == 4:
            node["edge_difference"] = 2
            # don't contract nodes with more than degree 4:
            # adds more than it takes away
        else:
            node["edge_difference"] = float("inf")
        node["importance_degree"] = calculate_importance_degree(node)

    def _check_connection_cost(self, node: int, neighbours: List[int]) -> bool:
        """a function to check the connection cost (i.e. travel cost) of going from a node to each of its neighbours:
        need to check how much battery is consumed from going from the first neighbour to any of the other ones
        Args:
            node (int)
            neighbours (List[int])

        Returns:
            bool
        """
        start = neighbours[0]
        cost_from_first_neighbour_to_node = self._get_edge_data(
            node, start, "battery_consumption"
        )
        for destinations in neighbours[1:]:
            dest_cost = self._get_edge_data(node, destinations, "battery_consumption")
            if dest_cost + cost_from_first_neighbour_to_node > 90:
                return False
        return True

    def _get_edge_data(
        self, u: int, v: int, data: str = "length"
    ) -> Dict[str, Union[str, int, bool]]:
        """returns a given edge's data

        Assumes that the key will always be 0

        Args:
            u (int)
            v (int)
            data (str, optional). Defaults to 'length'.

        Returns:
            Dict[Union[str, int, bool]]
        """
        return self.graph.get_edge_data(u, v)[0][data]

    def _get_neighbouring_nodes(self, node_id: int) -> List[int]:
        """returns all of the neighbours of a given node
        the same as using list(g.adj[node])
        Args:
            node_id (int)

        Returns:
            List[int]
        """
        return list(self.graph.neighbors(node_id))

    def _add_battery_consumption(self):
        """function to add battery consumtion attribute to all of the edges of the graph"""
        for edge in self.get_edges():
            edge[2]["battery_consumption"] = get_consumption_percentage(
                edge[2]["length"]
            )

    def _add_charging_stations(self):
        """function to add charging stations to a graph"""
        add_more = self._determine_additional_cs_number(self.charging_stations)
        selection_set = ListDict(self.times)
        self._add_cs_labels(self.charging_stations, add_more, selection_set)

    def _determine_additional_cs_number(self, node_approxs: List[int]) -> int:
        """function that determines the number of charging stations to be added in addition to the OSM ones, and any specified charging statiosn

        Args:
            node_approxs (List[int])

        Returns:
            int
        """
        target_cs = round(sqrt(self.data.shape[0])) * self.times
        add_more = 0
        if len(node_approxs) < target_cs:
            add_more = target_cs - len(node_approxs)
        return add_more

    def _add_cs_labels(
        self, charging_stations: List[int], add_more: int, selection_set: ListDict
    ):
        """Function to actually add the charging station label (CS) to each node in the graph

        Args:
            charging_stations (List[int])
            add_more (int)
            selection_set (ListDict)
        """
        for item in self.get_nodes():
            if item in charging_stations:
                self.graph.nodes[item]["CS"] = True
            else:
                if self.graph.degree(item) == 2:
                    # only randomly select from those with degree 2
                    selection_set.add_item(item)
                else:
                    self.graph.nodes[item]["CS"] = False
        while add_more:
            node = selection_set.choose_random_item()
            self.graph.nodes[node]["CS"] = True
            selection_set.remove_item(node)
            add_more -= 1
        for index in iter(selection_set.items):
            self.graph.nodes[index]["CS"] = False

    def __getitem__(self, index: int) -> Tuple[int, Dict[str, Union[int, str, bool]]]:
        """
        function that returns the node id and node data based on the node index, not node ID
        this is a way to make the nodes of the graph act as a python
        dictionary. So now we can do video_library[video_id] and it will
        return the node ID and node data if it exists ot throw a GraphException.
        See also: https://www.kite.com/python/answers/how-to-override-the-[]-operator-in-python
        """
        try:
            return self.data.loc[index]
        except KeyError:
            raise GraphException("Node does not exist")
