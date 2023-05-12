"""A application selection in the terminal"""
from matplotlib import pyplot as plt
from BiS4EV import bis_4_ev
from auxilary import construct_contraction_graph, expand_path, print_graph
from .Command_parser_location import CommandExceptionLocation, CommandParserLocation


def route_planner(object: CommandParserLocation):
    """Run BiS4EV"""
    run_bis4ev(object)


def run_bis4ev(object):
    processed_graph = object.graph.get_pre_processed_graph()
    contraction_graph = construct_contraction_graph(processed_graph)
    start = object.source_node
    target = object.target_node
    contraction_graph = determine_graph(
        processed_graph, contraction_graph, start, target
    )
    print(f"Attemping to find route from {object.origin} to {object.target}")
    planner = bis_4_ev(contraction_graph, start, target)
    route, charging_stops = planner.find_route()
    if route == -1:
        print("No route to show")
    else:
        if contraction_graph.name != "processed":
            route = expand_path(contraction_graph, object.graph, route)
        fig, ax = print_graph(contraction_graph, "route", route)
        plt.show()


def determine_graph(processed_graph, contraction_graph, start, target):
    try:
        contraction_graph.nodes[start]
        contraction_graph.nodes[target]
    except KeyError as e:
        # "expand" graph
        contraction_graph = processed_graph
    return contraction_graph
