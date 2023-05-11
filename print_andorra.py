from matplotlib import pyplot as plt
import osmnx as ox
from scipy import spatial
from pyrosm import OSM, get_data

from auxilary import construct_contraction_graph, construct_graph, print_graph


def print_charging_stations_glasgow(graph, pois, location):
    # can easy be modified to any other place: more to show the code
    scotland = get_data("scotland")
    osm = OSM(scotland)
    # glasgow OSM
    glasgow_osm = OSM(scotland, bounding_box=[-4.32, 55.83, -4.18, 55.89])
    # -4.37 55.81 -4.12 55.91
    ###
    custom_filter = {"amenity": ["charging_station"]}
    pois = glasgow_osm.get_pois(custom_filter=custom_filter)
    ###
    nodes, edges = glasgow_osm.get_network(nodes=True, network_type="driving")
    g = osm.to_graph(nodes, edges, graph_type="networkx")
    fig, ax = print_graph(graph)
    # actual location
    actual = [
        ax.scatter(pois.iloc[x]["lon"], pois.iloc[x]["lat"], c="black")
        for x in range(pois.shape[0])
    ][-1]
    # nearest (designated) node
    approx = [
        ax.scatter(
            nodes[
                nodes.id
                == ox.distance.nearest_nodes(
                    g, pois.iloc[x]["lon"], pois.iloc[x]["lat"]
                )
            ]["lon"].values[0],
            nodes[
                nodes.id
                == ox.distance.nearest_nodes(
                    g, pois.iloc[x]["lon"], pois.iloc[x]["lat"]
                )
            ]["lat"].values[0],
            c="red",
            marker=".",
        )
        for x in range(pois.shape[0])
    ][-1]
    a = [actual, approx]
    plt.legend(
        a,
        ["Actual Location", "Approximated node"],
        scatterpoints=1,
        loc="upper right",
        ncol=3,
        fontsize=12,
    )
    fig.savefig(location + "glasgow.png", bbox_inches="tight")


def print_charging_stations(graph, location):
    # print map with charging stations
    fig, ax = print_graph(graph)
    charging_nodes = [node for node in graph.nodes() if graph.nodes[node]["CS"]]
    points = [
        (graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in charging_nodes
    ]
    c = [ax.scatter(point[0], point[1], c="#001f3f", marker="x") for point in points][
        -1
    ]
    plt.legend(
        [c],
        ["Charging station"],
        scatterpoints=1,
        loc="upper right",
        ncol=3,
        fontsize=8,
    )
    # fig.savefig(location + "charging_stations_osm.png", bbox_inches='tight')


def print_contracted_nodes(pre_processed_graph, contracted_graph, location):
    contracted_nodes = []
    for node in pre_processed_graph.nodes():
        if node not in contracted_graph.nodes():
            contracted_nodes.append(
                (
                    pre_processed_graph.nodes[node]["x"],
                    pre_processed_graph.nodes[node]["y"],
                )
            )
    ## printing with original map
    # fig, ax = print_graph(pre_processed_graph)
    # a = [ax.scatter(point[0], point[1], c='#001f3f', s=3**2) for point in contracted_nodes][-1]
    # plt.legend([a],
    #         ["Contracted node"],
    #         scatterpoints=1,
    #         loc='upper right',
    #         fontsize=12)
    ## printing only contracted nodes
    plt.figure(figsize=(18, 18))
    a = [
        plt.scatter(point[0], point[1], c="#001f3f", s=3**2)
        for point in contracted_nodes
    ][-1]
    plt.legend(
        [a], ["Contracted node"], scatterpoints=1, loc="upper right", fontsize=12
    )
    plt.axis("off")
    # fig.savefig(location + "/non_contracted.png", bbox_inches='tight')


def convert_coords(graph, coords):
    # get the closest one
    distance, index = spatial.KDTree(graph.get_coords()).query(coords[::-1])
    place_node, node_data = graph[index]
    print(", index:", place_node)
    node_coords = (node_data["x"], node_data["y"])
    return node_coords


def extract_coords_from_locs(graph_object, place_addresses):
    coords = []
    for place in place_addresses:
        place_coords = ox.geocode(place)
        print("place:", place, end="")
        coords.append(convert_coords(graph_object, place_coords))
    return coords


def print_andorra(
    graph_object, graph, points_of_interest, location, include_legend=False
):
    points = []
    points = extract_coords_from_locs(graph_object, points_of_interest)
    fig, ax = print_graph(graph)
    markers = [
        "8",
        "v",
        "^",
        "<",
        ">",
        "s",
        "*",
        "$\diamond$",
        "P",
        "x",
        "D",
        "$\wedge$",
        "$\heartsuit$",
        "d",
        "$\dagger$",
        "$\leftrightarrow$",
    ]
    a = [
        ax.scatter(
            point[0],
            point[1],
            c="#001f3f",
            s=6**2,
            marker=markers[index % len(markers)],
        )
        for index, point in enumerate(points)
    ]
    if include_legend:
        plt.legend(a, places, scatterpoints=1, loc="upper right", ncol=3, fontsize=8)
    plt.show()
    # fig.savefig(location, bbox_inches='tight')


if __name__ == "__main__":
    places = [
        "Andorra la Vella",
        "Escaldes-Engordany",
        "Sant Julià de Lòria",
        "Encamp",
        "La Massana",
        "Ordino",
        "Canillo",
        "El Pas de la Casa",
        "Arinsal",
        "Bixessarri",
        "La Margineda",
        "Sispony",
        "Aubinyà",
        "Llorts",
        "Erts",
        "El Serrat",
    ]
    location = "./report_pics/"
    graph_object = construct_graph("Andorra", 1, 16)
    pre_processed_graph = graph_object.get_pre_processed_graph()
    contracted_graph = construct_contraction_graph(pre_processed_graph)
    # print_andorra(graph_object, contracted_graph, places, location, True)
    # print_charging_stations(pre_processed_graph, location)
