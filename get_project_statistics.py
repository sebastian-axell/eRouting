import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, kruskal, shapiro, norm
import seaborn as sns

easy_routes = [
    "('Encamp', 'Canillo')",
    "('Ordino', 'La Massana')",
    "('Bixessarri', 'Sant Julià de Lòria')",
    "('Arinsal', 'Ordino')",
    "('Escaldes-Engordany', 'Encamp')",
    "('Andorra La Vella', 'Sispony')",
    "('La Margineda', 'Aubinyà')",
    "('Llorts', 'Ordino')",
    "('Canillo', 'Ordino')",
    "('Andorra La Vella', 'Erts')",
]
hard_routes = ["('El Pas de la Casa', 'Arinsal')", "('El Serrat', 'Aubinyà')"]
b_label = "=========== BiS4EV ==========="
d_label = "=========== Dijkstra4EV ==========="
between_label = "=========== Between ==========="


def get_data(location, name):
    data = pd.read_excel(location)
    data["name"] = name
    return data


def print_metric_bell_curve(df, route, metric, location, name):
    route_df = df[df["route"] == route]
    stat, p = shapiro(route_df[metric])
    if p < 0.05:
        print(f"Route {route} metric {metric} is not normally distributed")
    sns.distplot(route_df[metric], fit=norm, kde=False).figure.savefig(
        f"location/{name}.png"
    )


def check_for_normal_distribution(df, routes, metrics):
    for route in routes:
        route_df = df[df["route"] == route]
        for metric in metrics:
            stat, p = shapiro(route_df[metric])
            if p < 0.05:
                print(f"Route {route} metric {metric} is not normally distributed")
            else:
                print("Data is normally distributed")


def get_stats(type):
    if type == "easy":
        b_results = get_results(bis4ev_data, "BiS4EV")
        d_results = get_results(dijkstra_data, "Dijkstra4EV")
        between = get_between_results(bis4ev_data, dijkstra_data)
        routes = easy_routes
    elif type == "hard":
        b_results = get_results(bis4ev_data_hard, "BiS4EV")
        d_results = get_results(dijkstra_data_hard, "Dijkstra4EV")
        between = get_between_results(bis4ev_data_hard, dijkstra_data_hard)
        routes = hard_routes
    for route in routes:
        print(route)
        print(b_label)
        get_table(b_results, route, ["name", "metric", "f", "p", "significant"])
        print(d_label)
        get_table(d_results, route, ["name", "metric", "f", "p", "significant"])
        print(between_label)
        get_table(between, route, ["metric", "f", "p", "significant"])


def check_variations(df, route, seed1, size1, seed2, size2):
    variation_results = pd.DataFrame(columns=columns[2:])
    index = 0
    route_df = df[df["route"] == route]
    data1 = route_df[(route_df["seed"] == seed1) & (route_df["size"] == size1)]
    data2 = route_df[(route_df["seed"] == seed2) & (route_df["size"] == size2)]
    for col in metrics:
        values1 = data1[col].values
        values2 = data2[col].values
        values = [values1[0], values2[0]]
        if len(set(values)) == 1:
            variation_results.at[index, "f"] = "N/A"
            variation_results.at[index, "p"] = "N/A"
            variation_results.at[index, "metric"] = col
            variation_results.at[index, "significant"] = False
        else:
            f, p_value = kruskal(*values)
            variation_results.at[index, "f"] = f
            variation_results.at[index, "p"] = p_value
            variation_results.at[index, "metric"] = col
            if p_value > 0.05:
                # print(f"Algorithm 1 is not significantly different from Algorithm 2 in for metric {col}")
                variation_results.at[index, "significant"] = False
            else:
                # print(f"Algorithm 1 is significantly different from Algorithm 2 in for metric {col}")
                variation_results.at[index, "significant"] = True
        index += 1
    return variation_results


def get_between_results(df1, df2):
    between_results = pd.DataFrame(columns=columns[1:])
    index = 0
    combined = pd.concat([df1, df2])
    for name, group in combined.groupby(["route"]):
        data1 = group[group["name"] == "bis4ev"]
        data2 = group[group["name"] == "dijkstra"]
        for col in metrics:
            values1 = data1[col].values
            values2 = data2[col].values
            if len(set(np.concatenate([values1, values2]))) == 1:
                between_results.at[index, "route"] = name
                between_results.at[index, "f"] = "N/A"
                between_results.at[index, "p"] = "N/A"
                between_results.at[index, "metric"] = col
                between_results.at[index, "significant"] = False
            else:
                f, p_value = wilcoxon(values1, values2)
                between_results.at[index, "route"] = name
                between_results.at[index, "f"] = f
                between_results.at[index, "p"] = p_value
                between_results.at[index, "metric"] = col
                if p_value > 0.05:
                    # print(f"Algorithm 1 is not significantly different from Algorithm 2 in for metric {col}")
                    between_results.at[index, "significant"] = False
                else:
                    # print(f"Algorithm 1 is significantly different from Algorithm 2 in for metric {col}")
                    between_results.at[index, "significant"] = True
            index += 1
    return between_results


def get_table(dataframe, route, values):
    print(dataframe[dataframe["route"] == route][values].to_latex(index=False))


def get_results(df1, given_name):
    df = pd.DataFrame(columns=columns)
    index = 0
    for name, group in df1.groupby(["route"]):
        df.at[index, "route"] = name
        for col in metrics:
            values = set(group[col].values)
            df.at[index, "route"] = name
            if len(values) == 1:
                # print(col, "is the same for all seed and sizes")
                df.at[index, "f"] = "N/A"
                df.at[index, "p"] = "N/A"
                df.at[index, "metric"] = col
                df.at[index, "significant"] = False
                df.at[index, "name"] = given_name
            else:
                f, pvalue = kruskal(*values)
                # print(f"{col} p-value: {pvalue:.4f} f value: {f:.4f}")
                df.at[index, "f"] = f
                df.at[index, "p"] = pvalue
                df.at[index, "metric"] = col
                df.at[index, "name"] = given_name
                if pvalue < 0.05:
                    df.at[index, "significant"] = True
                else:
                    df.at[index, "significant"] = False
            index += 1
    return df


def compare_metrics(route_level, route, metrics, details=False):
    if route_level == "hard":
        route_b = bis4ev_data_hard[bis4ev_data_hard["route"] == route]
        route_d = dijkstra_data_hard[dijkstra_data_hard["route"] == route]
    elif route_level == "easy":
        route_b = bis4ev_data[bis4ev_data["route"] == route]
        route_d = dijkstra_data[dijkstra_data["route"] == route]
    # makes sense
    for metric in metrics:
        print(metric)
        print(b_label)
        for name, group in route_b.groupby([metric]):
            print(name)
            print(group["seed"].values, group["size"].values)
            if details:
                print(group[metric].values)
        print(d_label)
        for name, group in route_d.groupby([metric]):
            print(name)
            print(group["seed"].values, group["size"].values)
            if details:
                print(group[metric].values)


def compare_metric_across(metric, include_details=False):
    print(b_label)
    for route in easy_routes:
        route_data = bis4ev_data[bis4ev_data["route"] == route]
        for name, group in route_data.groupby([metric]):
            if include_details:
                print(
                    name,
                    group["route"].values,
                    group["seed"].values,
                    group["size"].values,
                )
            else:
                print(name)
    for route in hard_routes:
        route_data = bis4ev_data_hard[bis4ev_data_hard["route"] == route]
        for name, group in route_data.groupby([metric]):
            if include_details:
                print(
                    name,
                    group["route"].values,
                    group["seed"].values,
                    group["size"].values,
                )
            else:
                print(name)
    print(d_label)
    for route in easy_routes:
        route_data = dijkstra_data[dijkstra_data["route"] == route]
        for name, group in route_data.groupby([metric]):
            if include_details:
                print(
                    name,
                    group["route"].values,
                    group["seed"].values,
                    group["size"].values,
                )
            else:
                print(name)
    for route in hard_routes:
        route_data = dijkstra_data_hard[dijkstra_data_hard["route"] == route]
        for name, group in route_data.groupby([metric]):
            if include_details:
                print(
                    name,
                    group["route"].values,
                    group["seed"].values,
                    group["size"].values,
                )
            else:
                print(name)


def print_precise_route_metrics(df, route):
    route_df = df[df["route"] == route]
    for metric in metrics:
        print(metric)
        for index, val in enumerate(route_df[metric].values):
            print(val)


if __name__ == "__main__":
    metrics = [
        "total_travel_time",
        "total_energy_cost",
        "total_distance",
        "total_charging_time",
    ]
    columns = ["name", "route", "metric", "f", "p", "significant"]
    # Easy
    dijkstra_data = get_data("./data/dijkstra/easy.xlsx", "dijkstra")
    bis4ev_data = get_data("./data/bis4ev/easy.xlsx", "bis4ev")
    # Hard
    dijkstra_data_hard = get_data("./data/dijkstra/hard.xlsx", "dijkstra")
    bis4ev_data_hard = get_data("./data/bis4ev/hard.xlsx", "bis4ev")
    # get_stats("easy")
    # compare_metrics("hard", "('El Pas de la Casa', 'Arinsal')", ['total_energy_cost', "total_distance"])
    # compare_metrics("easy", "('Andorra La Vella', 'Sispony')", ['total_energy_cost', 'total_distance'])
    # compare_metric_across("outcome", True)
    # compare_metrics("easy", "('Encamp', 'Canillo')", ['total_travel_time', 'total_charging_time'], True)
    # print_precise_route_metrics(bis4ev_data_hard, "('El Pas de la Casa', 'Arinsal')")
    # print(check_variations(bis4ev_data, "('Arinsal', 'Ordino')", 16, 0, 29, 3).to_latex(index=False))
