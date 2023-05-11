"""A command parser class for menu"""

import textwrap
from typing import Sequence


class CommandExceptionMenu(Exception):
    """A class used to represent a wrong command exception."""
    pass


class CommandParserMenu:
    """A class used to parse and execute a user command for the menu application."""

    def __init__(self):
        return

    def execute_command(self, command: Sequence[str]):
        """Executes the user command. Expects the command to be upper case.
           Raises CommandExceptionMenu if a command cannot be parsed.
        """
        if not command:
            raise CommandExceptionMenu(
                "Please enter a valid command, "
                "type HELP for a list of available commands.")
        
        # the defined commands for the menu application
        if command[0].upper() == "1":
            print("Starting BiS4EV Based Route Planner application...")

        elif command[0].upper() == "2":
            print("Coming soon...")  

        elif command[0].upper() == "HELP":
            self._get_help()    

        else:
            print(
                "Please enter a valid command, type HELP for a list of "
                "available commands.")



    def _get_help(self):
        """Displays all available commands to the user."""
        help_text = textwrap.dedent("""
        Available commands:
            1 - starts BiS4EV Based Route Planner
            2 - coming soon
            HELP - Displays help.
            EXIT - Terminates the program execution.
        """)
        print(help_text)
