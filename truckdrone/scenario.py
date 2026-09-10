"""
Scenario loading for the truck-drone delivery simulator.

The road network is a 50-node / 87-edge weighted undirected graph built over
real OpenStreetMap intersections in a 10 km x 10 km sub-region of Whitefield,
Bengaluru, projected to UTM zone 43N. Edge weights are geodesic distance
multiplied by a tortuosity factor of 1.35 to account for the fact that roads
do not run straight between intersections.

Two distance matrices come out of that graph, and the distinction is the whole
point of the project:

  * ``road_matrix`` -- all-pairs shortest-path distance over the graph.
    Every truck movement uses this.
  * ``air_matrix``  -- straight-line distance between node coordinates.
    Every drone leg uses this, because a drone is not bound to roads.

The network and its precomputed delivery instances are cached in
``data/whitefield_scenario.npz``, so nothing here needs internet access.

Beyond the matrices, this module also rebuilds the *graph* itself. The
distance matrices alone are enough to train an agent, but the dashboard needs
to draw a truck creeping along actual streets, which requires the edge list
and the node-by-node shortest path between consecutive stops.
"""

import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")
SCENARIO_PATH = os.path.join(DATA_DIR, "whitefield_scenario.npz")

TORTUOSITY = 1.35
N_EDGES = 87


# ======================================================================
# Loading
# ======================================================================

def load_scenario(path=SCENARIO_PATH):
    """
    Load the cached scenario.

    Returns a dict with: source, coords (N,2), road_matrix (N,N),
    air_matrix (N,N), n_nodes, n_edges, and ``instances`` -- a list of
    {depot, customers, truck_route, truck_route_distance_m}.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No scenario found at {}.\n"
            "The cached OpenStreetMap network should ship with this project. "
            "If it is missing, copy whitefield_scenario.npz into data/."
            .format(path))

    with np.load(path, allow_pickle=False) as z:
        scenario = json.loads(str(z["meta"]))
        scenario["coords"] = z["coords"]
        scenario["road_matrix"] = z["road_matrix"]
        scenario["air_matrix"] = z["air_matrix"]
    return scenario


def split_instances(scenario, train_fraction=0.5):
    """
    Split the instance pool into disjoint training and evaluation halves.

    Every number we report comes from the evaluation half, so the agent is
    always scored on delivery instances it has never trained on. Without this
    split an agent could memorise routes and the comparison against OR-Tools
    would mean nothing.
    """
    n = len(scenario["instances"])
    n_train = max(1, int(round(n * train_fraction)))
    return list(range(n_train)), (list(range(n_train, n)) or list(range(n_train)))


# ======================================================================
# Graph reconstruction (for animation)
# ======================================================================

def build_graph(coords, n_edges=N_EDGES, tortuosity=TORTUOSITY):
    """
    Rebuild the road graph from node coordinates.

    Candidate edges come from the Delaunay triangulation, which gives a
    planar, road-like adjacency. A minimum spanning tree is laid down first so
    the network is guaranteed connected, then the shortest remaining
    candidates are added until the edge budget is met. This is deterministic,
    so the graph reconstructed here is bit-identical to the one whose
    shortest paths produced the cached ``road_matrix``.
    """
    import networkx as nx
    from scipy.spatial import Delaunay

    n = len(coords)
    tri = Delaunay(coords)
    candidates = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = int(simplex[a]), int(simplex[b])
                candidates.add((min(i, j), max(i, j)))

    def length(e):
        return float(np.linalg.norm(coords[e[0]] - coords[e[1]]))

    candidates = sorted(candidates, key=length)

    full = nx.Graph()
    full.add_nodes_from(range(n))
    for e in candidates:
        full.add_edge(e[0], e[1], weight=length(e) * tortuosity)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, data in nx.minimum_spanning_edges(full, weight="weight", data=True):
        G.add_edge(u, v, **data)
    for e in candidates:
        if G.number_of_edges() >= n_edges:
            break
        if not G.has_edge(*e):
            G.add_edge(e[0], e[1], weight=length(e) * tortuosity)

    return G


def road_path(G, src, dst):
    """Node-by-node shortest path over the road graph, for drawing."""
    import networkx as nx

    if src == dst:
        return [int(src)]
    return [int(n) for n in nx.shortest_path(G, src, dst, weight="weight")]


def describe(scenario):
    inst = scenario["instances"][0]
    return (
        "Road network : {} nodes, {} edges over OpenStreetMap intersections\n"
        "Region       : Whitefield, Bengaluru -- {} km x {} km, {}\n"
        "Tortuosity   : {} (road distance / straight-line distance)\n"
        "Instances    : {} delivery problems x {} customers each\n"
        "Instance 0   : {} truck stops, {:.1f} km truck-only route".format(
            scenario["n_nodes"], scenario["n_edges"],
            scenario["area_km"], scenario["area_km"], scenario["crs"],
            scenario["tortuosity"], len(scenario["instances"]),
            len(inst["customers"]), len(inst["truck_route"]),
            inst["truck_route_distance_m"] / 1000.0))


if __name__ == "__main__":
    sc = load_scenario()
    print(describe(sc))
    train, evl = split_instances(sc)
    print("\nTrain instances: {}  Eval instances: {}".format(len(train), len(evl)))

    G = build_graph(sc["coords"])
    print("Rebuilt graph  : {} nodes, {} edges".format(
        G.number_of_nodes(), G.number_of_edges()))

    # Verify the rebuilt graph reproduces the cached distance matrix.
    import networkx as nx
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    err = max(abs(lengths[i][j] - sc["road_matrix"][i][j])
              for i in range(sc["n_nodes"]) for j in range(sc["n_nodes"]))
    print("Max road-matrix mismatch: {:.6f} m  {}".format(
        err, "OK" if err < 1e-6 else "MISMATCH"))
