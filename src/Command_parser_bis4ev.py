"""A command parser class for BiS4EV planner"""

import textwrap
from typing import Sequence


class CommandExceptionBiS4EV(Exception):
    """A class used to represent a wrong command exception."""
    pass


class CommandParser:
    """A class used to parse and execute a user command for BiS4EV planner class."""

    def __init__(self, planner):
        self.planner = planner

    def execute_command(self, command: Sequence[str]):
        """Executes the user command. Expects the command to be upper case.
           Raises CommandExceptionBiS4EV if a command cannot be parsed.
        """
        # the defined commands for Hashplanner
        if not command:
            raise CommandExceptionBiS4EV(
                "Please enter a valid command, "
                "type HELP for a list of available commands.")

        if command[0].lower() == "find_route":
            self.planner.find_route()
        
        elif command[0].lower() == "test":
            self.planner.test()

        elif command[0].lower() == "help":
            self._get_help()
        else:
            print(
                "Please enter a valid command, type HELP for a list of "
                "available commands.")

    def _get_help(self):
        """Displays all available commands to the user."""
        help_text = textwrap.dedent("""
        Available commands:
            HELP - Displays help.
            EXIT - Terminates the program execution.
            FIND_ROUTE - Takes an origin and a destination and finds a route
        """)
        print(help_text)
