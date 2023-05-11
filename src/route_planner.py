"""A application selection in the terminal"""
from auxilary import construct_contraction_graph, expand_path
from .Command_parser_menu import CommandExceptionMenu, CommandParserMenu
from .Command_parser_location import CommandParserLocation, CommandExceptionLocation

def route_planner(object: CommandParserLocation): 
    """Defines the application interface for the route planner selection"""
    print("""Hello and welcome to route planner selection. 
    Please select planner 1 (BiS4EV Based Route Planner) or planner 2 (coming soon) by entering 1 or 2 respectively.""")
    parser = CommandParserMenu()
    while True:
        command = input("Select Route Planner> ")
        if command.upper() == "EXIT":
            break
        try:
            parser.execute_command(command.split())
            if command.split()[0] == "1":
                processed_graph = object.graph.get_pre_processed_graph()
                contraction_graph = construct_contraction_graph(processed_graph)
                start = object.source_node
                target = object.target_node
                try:
                    contraction_graph.nodes[start]
                    contraction_graph.nodes[target]
                except KeyError as e:
                    # "expand" graph
                    print("FUCK")
                    return
                    contraction_graph =  processed_graph
                print(f"Attemping to find route from {object.origin} to {object.target}")
                print(f"{start} to {target}")
                planner = bis_4_ev(contraction_graph)
                route, data1, data2, data3 = planner.find_route(start, target)
                print(route)
                expand_path(contraction_graph, route)
                planner.print_graph()
                print("Done")
            if command.split()[0] == "2":
                pass  
        except Exception as e:
            raise CommandExceptionLocation(e)
    print("Application has now terminated its execution. "
        "Thank you and goodbye!")