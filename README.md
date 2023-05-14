# eRouting
The code used for my honours project which involved evaluating the quality of routes produced by a recently proposed routing algorithm for electric vehicles, BiS4EV.

src folder contains menu application to try out BiS4EV. To run the menu application, use python -m src.run. The user manual contains the instructions on how to use the other source code.

The following dependencies are needed:
• textwrap
• scipy.stats
• seaborn
• matplotlib
• time
• re
• os
• matplotlib.pyplot
• pyrosm
• typing
• math
• networkx
• pandas
• numpy
• heapq
• osmnx
• random
• pyrosm.data
• scipy

The submission contains the following files:

Root directory
    maintenance_manual.pdf
    auxilary.py
    BiS4EV.py
    Dijkstra4EV.py
    get_project_statistics.py
    graph.py
    ListDict.py
    path.py
    print_andorra.py
    print_paths.py
    README.md
    testing_suite.py
    pics
    report_pics
    data
        bis4ev
            easy.xlsx
            hard.xlsx
        dijkstra
            easy.xlsx
            hard.xlsx
    src
        Command_parser_location.py
        route_planner.py
        run.py
        __init__.py
    paths
        bis4ev
            bis4ev_Andorra La Vella_Erts_16_0_easy.txt
            bis4ev_Andorra La Vella_Sispony_16_0_easy.txt
            bis4ev_Arinsal_Ordino_16_0_easy.txt
            bis4ev_Arinsal_Ordino_29_3_easy.txt
            bis4ev_Bixessarri_Sant Julia de Loria_16_0_easy.txt
            bis4ev_Canillo_Ordino_16_0_easy.txt
            bis4ev_El Pas de la Casa_Arinsal_16_0_hard.txt
            bis4ev_El Pas de la Casa_Arinsal_29_3_hard.txt
            bis4ev_El Serrat_Aubinya_16_0_hard.txt
            bis4ev_Encamp_Canillo_16_0_easy.txt
            bis4ev_Encamp_Canillo_4_3_easy.txt
            bis4ev_Escaldes-Engordany_Encamp_16_0_easy.txt
            bis4ev_La Margineda_Aubinya_16_0_easy.txt
            bis4ev_Llorts_Ordino_16_0_easy.txt
            bis4ev_Ordino_La Massana_16_0_easy.txt
        dijkstra
            dijkstra_Andorra La Vella_Sispony_16_0_easy.txt

The code in its entirety can be found here: https://github.com/sebastian-axell/eRouting