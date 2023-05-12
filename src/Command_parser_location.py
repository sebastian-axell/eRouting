"""A command parser class for location selection"""

import textwrap
from typing import Sequence
from pyrosm.data import sources
from auxilary import construct_graph
from graph import GraphException
import osmnx as ox
from scipy import spatial


def display_datasets(dictionary):
    intro = ""
    for key, values in dictionary.items():
        if isinstance(values, dict):
            intro += f"\n====== {'->' + key +'<-' :>30} {'======' :>30}\n"
            intro += display_datasets(values)
            continue
        intro += f"\n=== {'->' + key +'<-' :>30} {'===' :>30}\n"
        if len(values) == 1:
            intro += values[0] + "\n"
        else:
            for a, b, c in zip(values[::3], values[1::3], values[2::3]):
                intro += "{:<30}{:<30}{:<}\n".format(a, b, c)
            intro += "\n"
    return intro


class CommandExceptionLocation(Exception):
    """A class used to represent a wrong command exception."""

    pass


class CommandParserLocation:
    """A class used to parse and execute a user command for the menu application."""

    def __init__(self):
        self.graph = None
        self.source_node = None
        self.target_node = None
        self.region = None
        self.origin = None
        self.target = None
        self.coords = None

    def __eq__(self, __o: object) -> bool:
        return None not in [
            self.graph,
            self.source_node,
            self.target_node,
            self.origin,
            self.target,
        ]

    def __str__(self) -> str:
        return f"{self.target}, {self.graph}, {self.origin}"

    def execute_command(self, command: Sequence[str]):
        """Executes the user command.
        Raises CommandExceptionLocation if a command cannot be parsed.
        """
        if not command:
            raise CommandExceptionLocation(
                "Please enter a valid command, "
                "type HELP for a list of available commands."
            )
        command = command.lower()
        if command == "help":
            self._get_help()
        elif command == "region":
            self._region_select()
        elif command in ["origin", "target"]:
            self._place_select(command)
        elif command == "choices":
            self._choice()
        elif command == "continue":
            return self == True
        else:
            print(
                "Please enter a valid command, type help for a list of "
                "available commands."
            )

    def _region_select(self):
        graph = self.graph  # the value if not none or none
        location = self.region  # the value if not none or none
        print(
            f"""Please select your region, i.e. country or city, by entering a search query"""
        )
        self._get_help("region")
        while True:
            command = input("Region Select> ").lower()
            if command == "back":
                break
            if command == "help":
                self._get_help("region")
                continue
            if command == "places":
                print(display_datasets(sources.available))
                continue
            location = command
            try:
                graph = construct_graph(location)
                self.region = location
                self.graph = graph
                self.coords = graph.get_coords()
                break
            except ValueError as e:
                raise CommandExceptionLocation(
                    f"{command} is not a valid dataset. Use command 'places' to see the available datasets."
                )
            except Exception as e:
                raise CommandExceptionLocation(e)

    def _place_select(self, type):
        if not self.region:
            raise CommandExceptionLocation(
                "No region selected: please select region first"
            )
        place_address = (
            self.origin if type == "origin" else self.target
        )  # the current value or none
        place_node = (
            self.source_node if type == "origin" else self.target_node
        )  # the current value or none
        region = self.region
        print(
            f"""Please select your {type} by entering a search query.
            Current region: {region}"""
        )
        self._get_help("place")
        while True:
            command = (
                input("Origin Select> ").lower()
                if type == "origin"
                else input("Target Select> ").lower()
            )
            if command == "back":
                break
            if command == "help":
                self._get_help("place")
                continue
            place_address = command
            try:
                place_coords = ox.geocode(place_address)
                place_node, node_coords = self.convert_coords(place_coords)
                if type == "origin":
                    self.set_origin_values(place_address, place_node)
                else:
                    self.set_target_values(place_address, place_node)
                break
            except GraphException as ge:
                raise CommandExceptionLocation(ge)
            except Exception as e:
                raise CommandExceptionLocation(e)

    def convert_coords(self, place_coords):
        distance, index = spatial.KDTree(self.coords).query(place_coords[::-1])
        place_node, node_data = self.graph[index]
        node_coords = (node_data["x"], node_data["y"])
        return place_node, node_coords

    def set_target_values(self, place_address, place_node):
        self.target = place_address
        self.target_node = place_node
        if self.origin == self.target or self.source_node == self.target_node:
            self.target = None
            self.target_node = None
            raise CommandExceptionLocation(
                "Origin and target cannot be the same; please select target again."
            )

    def set_origin_values(self, place_address, place_node):
        self.origin = place_address
        self.source_node = place_node
        if self.origin == self.target or self.source_node == self.target_node:
            self.origin = None
            self.source_node = None
            raise CommandExceptionLocation(
                "Origin and target cannot be the same; please select origin again."
            )

    def _get_help(self, type="general"):
        """Displays all available commands to the user."""
        help_text_general = textwrap.dedent(
            """
        Available commands:
            region - Starts region selector
            origin - Starts origin selector
            target - Starts target selector
            help - Displays this help menu.
            exit - Terminates the program execution.
            choices - Displays the current choices
        """
        )
        help_text_region = textwrap.dedent(
            """
        Available commands:
            back - Go back to main menu.
            help - Displays this help menu.
            places - Displays available datasets.
        
        Other entries will be interpreted as search queries.
        """
        )
        help_text_place = textwrap.dedent(
            """
        Available commands:
            back - Go back to main menu.
            help - Displays this help menu.
        
        Other entries will be interpreted as search queries.
        """
        )
        if type == "region":
            print(help_text_region)
        elif type == "place":
            print(help_text_place)
        else:
            print(help_text_general)

    def _choice(self):
        """Displays the current choices by the user."""
        help_text = textwrap.dedent(
            f"""
        Current choices:
            region - {self.region}
            origin - {self.origin}
            target - {self.target}
        
        To change these select their respective command from the main menu.
        """
        )
        print(help_text)
