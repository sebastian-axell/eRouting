"""A menu application in the terminal"""
from .Command_parser_location import CommandExceptionLocation, CommandParserLocation
from .route_planner import route_planner
import os


if __name__ == "__main__": 
    """Defines the application interface for the menu"""
    print("""Hello and welcome to this commandline application. 
        Please select a choice """)
    parser = CommandParserLocation()
    parser._get_help()
    while True:
        target = None
        command = input("Menu> ")
        if command.lower() == "exit":
            break
        try:
            parser.execute_command(command)
        except CommandExceptionLocation as e:
            print(e)
        if parser.execute_command("continue"):
            parser._choice()
            if input("Continue to route planner? [y,n]: ").lower() in ['yes', 'y']:
                os.system('cls')
                route_planner(parser)
                parser = CommandParserLocation()
    print(parser)
    print("Application has now terminated its execution. "
        "Thank you and goodbye!")