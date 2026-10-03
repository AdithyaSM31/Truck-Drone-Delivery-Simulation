"""
Scenarios: real road networks, delivery instances, and the street geometry
needed to draw them.

WHAT CHANGED IN VERSION 2
-------------------------
Version 1 sampled 50 real OpenStreetMap intersections and then joined them with
*synthetic* edges -- a Delaunay triangulation with every straight line inflated
by a 1.35 "tortuosity" factor. The intersections were real; the roads between
them were not. That was defensible for training, but it meant the truck drove
through buildings on any real map, ignored one-way streets, and every distance
the results quoted rested on one assumed constant.

Version 2 drives on the actual street network. The 50 sites are the same real
intersections, but the truck's distance from one to another is now the shortest
path through the full OpenStreetMap drive graph -- about ten thousand
intersections and twenty thousand directed street segments, one-way streets
included. Distances are therefore *asymmetric*: A to B is not B to A when a
one-way system is in the way. The 1.35 factor is gone; the tortuosity is now
measured, and reported per city.

Two matrices still carry the whole idea:

  * ``road_matrix`` -- directed shortest-path driving distance. The truck.
  * ``air_matrix``  -- straight-line distance. The drone, which ignores roads.

CITIES
------
``whitefield`` is the training city (an IT corridor on the eastern edge of
Bengaluru). ``chennai`` is a second, structurally different city (dense,
older street grid around T. Nagar) used only to test whether a policy trained
in one place transfers to another without retraining.

FILES (all under data/)
-----------------------
  <city>_drive.graphml    the raw OpenStreetMap extract (WGS84)
  <city>_scenario.npz     matrices, coordinates, delivery instances
  <city>_paths.npz        street polyline for every ordered pair of sites,
                          plus a background of major roads, for drawing

The simulator itself only reads the scenario file. Paths are needed only by
the dashboard and the Folium maps.

    python -m truckdrone.scenario --build whitefield
    python -m truckdrone.scenario --build chennai
"""

import argparse
import json
import os
import time

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data")

CITIES = {
    "whitefield": {
        "name": "Whitefield, Bengaluru",
        "lat": 12.9698, "lon": 77.7499, "area_km": 10.0,
        "crs": "EPSG:32643",           # UTM zone 43N
        "sample_seed": 42,
    },
    "chennai": {
        "name": "T. Nagar, Chennai",
        "lat": 13.0418, "lon": 80.2341, "area_km": 10.0,
        "crs": "EPSG:32644",           # UTM zone 44N
        "sample_seed": 7,
    },
}
DEFAULT_CITY = "whitefield"

N_SITES = 50
MAIN_CUSTOMERS = 15
N_MAIN_INSTANCES = 40          # split 20 train / 20 held-out evaluation
EXTRA_COUNTS = (10, 20)        # held-out only: does the policy generalise?
N_EXTRA_INSTANCES = 20
MIN_SPACING_M = 600.0
BACKGROUND_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary",
                       "motorway_link", "trunk_link", "primary_link",
                       "secondary_link", "tertiary_link"}

# Overpass mirrors. The default endpoint is unreachable from some networks
# that otherwise have full internet access; an env var lets a user pin one.
_OVERPASS = [u for u in (os.environ.get("OSMNX_OVERPASS_URL"),
                         "https://overpass-api.de/api",
                         "https://lz4.overpass-api.de/api",
                         "https://overpass.kumi.systems/api") if u]


def graphml_path(city):
    return os.path.join(DATA_DIR, "{}_drive.graphml".format(city))


def scenario_path(city):
    return os.path.join(DATA_DIR, "{}_scenario.npz".format(city))


def paths_path(city):
    return os.path.join(DATA_DIR, "{}_paths.npz".format(city))


# ======================================================================
# Raw network
# ======================================================================

def _bbox(lat, lon, area_km):
    half = area_km / 2.0
    dlat = half / 111.32
    dlon = half / (111.32 * np.cos(np.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def load_osm_graph(city):
    """
    The OpenStreetMap drive network for ``city`` in WGS84, downloaded once and
    cached as GraphML. Only the largest strongly connected component is kept:
    a delivery site the truck can reach but never leave (or vice versa) is not
    a site a truck can serve.
    """
    import networkx as nx
    import osmnx as ox

    spec = CITIES[city]
    path = graphml_path(city)
    if os.path.exists(path):
        G = ox.load_graphml(path)
    else:
        west, south, east, north = _bbox(spec["lat"], spec["lon"], spec["area_km"])
        errors, G = [], None
        for endpoint in _OVERPASS:
            ox.settings.overpass_url = endpoint
            try:
                G = ox.graph_from_bbox((west, south, east, north),
                                       network_type="drive")
                break
            except Exception as exc:                       # noqa: BLE001
                errors.append("{}: {}".format(endpoint, exc))
        if G is None:
            raise RuntimeError("Could not download {} from OpenStreetMap:\n  {}"
                               .format(city, "\n  ".join(errors)))
        os.makedirs(DATA_DIR, exist_ok=True)
        ox.save_graphml(G, path)

    scc = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(scc).copy()


def _transformer(crs, inverse=False):
    from pyproj import Transformer
    if inverse:
        return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def to_utm(lonlat, crs):
    lonlat = np.asarray(lonlat, dtype=np.float64)
    x, y = _transformer(crs).transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


def to_lonlat(xy, crs):
    xy = np.asarray(xy, dtype=np.float64)
    lon, lat = _transformer(crs, inverse=True).transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lon, lat])


# ======================================================================
# Site selection and instances
# ======================================================================

def _farthest_point_sample(xy, k, seed):
    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(len(xy)))]
    d = np.linalg.norm(xy - xy[picked[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(xy - xy[nxt], axis=1))
    return picked


def _sample_customers(coords, n_customers, seed, depot):
    """n_customers well-spread sites other than the depot."""
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(coords)) if int(i) != depot]
    spacing = MIN_SPACING_M
    while spacing >= 1.0:
        chosen = []
        for cand in order:
            if len(chosen) >= n_customers:
                break
            if all(np.linalg.norm(coords[cand] - coords[c]) >= spacing
                   for c in chosen):
                chosen.append(cand)
        if len(chosen) >= n_customers:
            return chosen[:n_customers]
        spacing *= 0.7
    raise RuntimeError("Cannot place {} customers.".format(n_customers))


def _solve_tour(road_matrix, depot, customers, time_limit_s=2):
    """OR-Tools tour over {depot} + customers on the directed road matrix."""
    from .baselines import solve_tsp

    nodes = [depot] + list(customers)
    sub = road_matrix[np.ix_(nodes, nodes)]
    t0 = time.perf_counter()
    route, dist = solve_tsp(sub, depot=0, time_limit_s=time_limit_s)
    return ([int(nodes[i]) for i in route], float(dist),
            time.perf_counter() - t0)


def _instances(road_matrix, coords, depot, counts_and_seeds):
    out = []
    for n_customers, seed in counts_and_seeds:
        customers = _sample_customers(coords, n_customers, seed, depot)
        route, dist, solve_s = _solve_tour(road_matrix, depot, customers)
        out.append({"depot": depot, "customers": customers,
                    "truck_route": route, "truck_route_distance_m": dist,
                    "ortools_solve_s": solve_s})
    return out


# ======================================================================
# Street geometry
# ======================================================================

def _edge_points(G, u, v):
    """Coordinates along the shortest parallel edge u->v, curves included."""
    data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0.0))
    geom = data.get("geometry")
    if geom is not None:
        return list(geom.coords)
    return [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]


def _node_path_to_lonlat(G, path):
    pts = [(G.nodes[path[0]]["x"], G.nodes[path[0]]["y"])]
    for u, v in zip(path[:-1], path[1:]):
        pts.extend(_edge_points(G, u, v)[1:])
    return pts


def _background(G):
    """Major roads only: the backdrop the dashboard draws under everything."""
    lines = []
    for u, v, d in G.edges(data=True):
        hw = d.get("highway")
        hw = hw if isinstance(hw, list) else [hw]
        if not BACKGROUND_HIGHWAYS.intersection(h for h in hw if h):
            continue
        if u > v and G.has_edge(v, u):     # draw two-way roads once
            continue
        geom = d.get("geometry")
        lines.append(list(geom.coords) if geom is not None else
                     [(G.nodes[u]["x"], G.nodes[u]["y"]),
                      (G.nodes[v]["x"], G.nodes[v]["y"])])
    return lines


def _pack_polylines(polys):
    """Ragged list of point lists -> (offsets, float32 points) for npz."""
    offsets = np.zeros(len(polys) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in polys])
    points = (np.concatenate([np.asarray(p, dtype=np.float64) for p in polys])
              if polys else np.zeros((0, 2)))
    return offsets, points


def _unpack_polylines(offsets, points):
    return [points[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# ======================================================================
# Build
# ======================================================================

def build_scenario(city, reuse_sites_from=None):
    """
    Build and cache a city's scenario and street paths.

    ``reuse_sites_from`` keeps the 50 delivery sites -- and every instance's
    depot and customers -- from an earlier version-1 scenario file. Whitefield
    is rebuilt that way, so the forty delivery problems are the same problems
    as before; only the distances between their stops changed, from an assumed
    tortuosity to the real street network.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    spec = CITIES[city]
    t0 = time.perf_counter()
    G = load_osm_graph(city)
    osm_ids = list(G.nodes)
    lonlat_all = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in osm_ids])
    xy_all = to_utm(lonlat_all, spec["crs"])
    print("[{}] {} intersections, {} directed street segments ({:.0f}s)".format(
        city, G.number_of_nodes(), G.number_of_edges(), time.perf_counter() - t0))

    legacy = None
    if reuse_sites_from and os.path.exists(reuse_sites_from):
        with np.load(reuse_sites_from, allow_pickle=False) as z:
            legacy = {"coords": z["coords"], "meta": json.loads(str(z["meta"]))}

    if legacy is not None:
        # Snap each legacy site to its intersection in the connected network.
        dist, idx = cKDTree(xy_all).query(legacy["coords"])
        site_idx = [int(i) for i in idx]
        print("[{}] reused {} legacy sites (worst snap {:.1f} m)".format(
            city, len(site_idx), float(dist.max())))
    else:
        site_idx = _farthest_point_sample(xy_all, N_SITES, spec["sample_seed"])

    sites = [osm_ids[i] for i in site_idx]
    coords = xy_all[site_idx]
    lonlat = lonlat_all[site_idx]

    # Directed driving distance and the street path for every ordered pair.
    n = len(sites)
    road = np.zeros((n, n))
    pair_polys, pair_index = [], []
    for i, s in enumerate(sites):
        pred, dist = nx.dijkstra_predecessor_and_distance(G, s, weight="length")
        for j, t in enumerate(sites):
            if i == j:
                continue
            road[i, j] = dist[t]
            path, node = [t], t
            while node != s:
                node = pred[node][0]
                path.append(node)
            pair_polys.append(_node_path_to_lonlat(G, path[::-1]))
            pair_index.append((i, j))
    air = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    off = ~np.eye(n, dtype=bool)
    tortuosity = road[off] / np.maximum(air[off], 1.0)
    asym = np.abs(road - road.T)[off] / np.maximum(road[off], 1.0)
    print("[{}] tortuosity mean {:.3f} (median {:.3f}); one-way asymmetry "
          "affects {:.0f}% of pairs".format(city, tortuosity.mean(),
                                             np.median(tortuosity),
                                             100 * (asym > 0.01).mean()))

    centroid = coords.mean(axis=0)
    depot = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))

    if legacy is not None:
        main = []
        for inst in legacy["meta"]["instances"]:
            route, dist, solve_s = _solve_tour(road, inst["depot"], inst["customers"])
            main.append({"depot": inst["depot"], "customers": inst["customers"],
                         "truck_route": route, "truck_route_distance_m": dist,
                         "ortools_solve_s": solve_s})
        depot = main[0]["depot"]
    else:
        main = _instances(road, coords, depot,
                          [(MAIN_CUSTOMERS, spec["sample_seed"] + i)
                           for i in range(N_MAIN_INSTANCES)])
    extra = {str(k): _instances(road, coords, depot,
                                [(k, 1000 * k + i) for i in range(N_EXTRA_INSTANCES)])
             for k in EXTRA_COUNTS}

    meta = {
        "version": 2, "city": city, "name": spec["name"], "source": "osm",
        "crs": spec["crs"], "centre_lat": spec["lat"], "centre_lon": spec["lon"],
        "area_km": spec["area_km"], "n_nodes": n,
        "network_intersections": G.number_of_nodes(),
        "network_segments": G.number_of_edges(),
        "tortuosity_mean": float(tortuosity.mean()),
        "tortuosity_median": float(np.median(tortuosity)),
        "asymmetric_pair_fraction": float((asym > 0.01).mean()),
        "osm_ids": [str(s) for s in sites],
        "instances": main, "extra_instances": extra,
    }
    np.savez_compressed(scenario_path(city), coords=coords, lonlat=lonlat,
                        road_matrix=road, air_matrix=air, meta=json.dumps(meta))

    p_off, p_pts = _pack_polylines(pair_polys)
    b_off, b_pts = _pack_polylines(_background(G))
    np.savez_compressed(paths_path(city), pair_index=np.array(pair_index),
                        pair_offsets=p_off, pair_points=p_pts.astype(np.float32),
                        bg_offsets=b_off, bg_points=b_pts.astype(np.float32))
    print("[{}] built in {:.0f}s -> {}".format(city, time.perf_counter() - t0,
                                              scenario_path(city)))
    return load_scenario(city)


# ======================================================================
# Observation embedding
# ======================================================================

def embed_positions(road_matrix):
    """
    2-D site positions from multidimensional scaling of the road-distance
    matrix, normalised to [-1, 1], so Euclidean distance in the observation
    approximates *driving* distance. Real streets are directed, so the matrix
    is symmetrised first: MDS needs a dissimilarity, and the mean of the two
    directions is the honest single number for "how far apart".

    It depends only on the street network, so it is computed once and stored
    with the scenario. Training workers then never import scikit-learn -- which
    matters, because each extra BLAS-backed library a worker loads costs real
    memory times the number of workers.
    """
    import warnings
    from sklearn.manifold import MDS

    d = 0.5 * (road_matrix + road_matrix.T)
    finite = d[np.isfinite(d)]
    d[~np.isfinite(d)] = float(finite.max()) * 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                 normalized_stress="auto", n_init=4).fit_transform(d)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    return 2.0 * (xy - lo) / np.where(hi - lo == 0, 1.0, hi - lo) - 1.0


def _ensure_embedding(path):
    """Add ``node_xy`` to a scenario file that predates it (one-time)."""
    with np.load(path, allow_pickle=False) as z:
        if "node_xy" in z:
            return
        arrays = {k: z[k] for k in z.files}
    arrays["node_xy"] = embed_positions(arrays["road_matrix"])
    np.savez_compressed(path, **arrays)


# ======================================================================
# Load
# ======================================================================

def load_scenario(city=DEFAULT_CITY, path=None):
    """
    Load a cached scenario: coords (N,2) UTM metres, lonlat, road_matrix
    (directed), air_matrix, and ``instances`` -- the 15-customer problems,
    each with depot, customers and an OR-Tools truck tour. ``extra_instances``
    holds held-out problems with other customer counts.
    """
    path = path or scenario_path(city)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No scenario at {}. Build it:\n    python -m truckdrone.scenario "
            "--build {}".format(path, city))
    _ensure_embedding(path)
    with np.load(path, allow_pickle=False) as z:
        sc = json.loads(str(z["meta"]))
        for key in ("coords", "road_matrix", "air_matrix", "node_xy"):
            sc[key] = z[key]
        if "lonlat" in z:
            sc["lonlat"] = z["lonlat"]
    sc.setdefault("city", city)
    return sc


def with_instances(scenario, instances):
    """A shallow copy of ``scenario`` whose instance list is replaced."""
    out = dict(scenario)
    out["instances"] = instances
    return out


def split_instances(scenario, train_fraction=0.5):
    """
    Disjoint training and evaluation halves. Every reported number comes from
    the evaluation half, so the agent is scored on delivery problems it never
    trained on.
    """
    n = len(scenario["instances"])
    n_train = max(1, int(round(n * train_fraction)))
    return list(range(n_train)), (list(range(n_train, n)) or list(range(n_train)))


class StreetPaths:
    """Street polylines between sites, in UTM metres, for drawing."""

    def __init__(self, city=DEFAULT_CITY, crs=None):
        crs = crs or CITIES[city]["crs"]
        with np.load(paths_path(city), allow_pickle=False) as z:
            pairs = z["pair_index"]
            polys = _unpack_polylines(z["pair_offsets"], z["pair_points"])
            background = _unpack_polylines(z["bg_offsets"], z["bg_points"])
        self.crs = crs
        self._lonlat = {(int(a), int(b)): p for (a, b), p in zip(pairs, polys)}
        self._xy = {}
        self.background_lonlat = background

    def lonlat(self, a, b):
        if a == b:
            return None
        return self._lonlat[(int(a), int(b))]

    def xy(self, a, b):
        key = (int(a), int(b))
        if key not in self._xy:
            self._xy[key] = to_utm(self._lonlat[key], self.crs)
        return self._xy[key]

    def background_xy(self):
        return [to_utm(p, self.crs) for p in self.background_lonlat]


def describe(scenario):
    inst = scenario["instances"][0]
    return (
        "City         : {} ({})\n"
        "Street graph : {:,} intersections, {:,} directed segments (OpenStreetMap)\n"
        "Sites        : {} delivery sites; depot + {} customers per instance\n"
        "Tortuosity   : {:.2f} mean road/air ratio, measured; {:.0f}% of site "
        "pairs asymmetric (one-way streets)\n"
        "Instances    : {} main + {}\n"
        "Instance 0   : truck-only tour {:.1f} km".format(
            scenario.get("name", scenario["city"]), scenario["crs"],
            scenario.get("network_intersections", 0),
            scenario.get("network_segments", 0), scenario["n_nodes"],
            len(inst["customers"]), scenario.get("tortuosity_mean", float("nan")),
            100 * scenario.get("asymmetric_pair_fraction", 0.0),
            len(scenario["instances"]),
            ", ".join("{} x {} customers".format(len(v), k)
                      for k, v in scenario.get("extra_instances", {}).items()),
            inst["truck_route_distance_m"] / 1000.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", choices=list(CITIES))
    ap.add_argument("--city", default=DEFAULT_CITY, choices=list(CITIES))
    args = ap.parse_args()
    if args.build:
        legacy = os.path.join(DATA_DIR, "whitefield_scenario_v1.npz")
        sc = build_scenario(args.build, reuse_sites_from=(
            legacy if args.build == "whitefield" else None))
    else:
        sc = load_scenario(args.city)
    print()
    print(describe(sc))


if __name__ == "__main__":
    main()
