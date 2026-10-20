import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
import networkx as nx
from tqdm import tqdm
import sys

# helper.py and readNetwork.py live in the parent folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from helper import convert_network_tntp_to_dat, convert_demand_tntp_to_dat
from readNetwork import Network


# Worker-global state, populated by _init_worker once per process
_BASE_GRAPH = None
_CENTROID_OUT_EDGES = None
_FIRST_THRU = None


def _init_worker(edges, first_thru_node):
    """Initialize worker process with a shared base graph and centroid edge map."""
    global _BASE_GRAPH, _CENTROID_OUT_EDGES, _FIRST_THRU
    _FIRST_THRU = first_thru_node

    base = nx.DiGraph()
    centroid_out = defaultdict(list)

    for u, v, w in edges:
        u_i = int(u)
        v_i = int(v)
        if u_i < first_thru_node:
            # Outgoing edge from a centroid, stash for per-origin reinstatement
            centroid_out[u_i].append((v_i, w))
        else:
            base.add_edge(u_i, v_i, weight=w)

    _BASE_GRAPH = base
    _CENTROID_OUT_EDGES = centroid_out


def process_origin(args):
    """Worker: compute k shortest paths from orgn to each dest."""
    orgn, dests, k = args
    orgn = int(orgn)
    dests = [int(d) for d in dests]

    # Copy base graph and add back only the current origin's outgoing edges
    G = _BASE_GRAPH.copy()
    for v, w in _CENTROID_OUT_EDGES.get(orgn, []):
        G.add_edge(orgn, v, weight=w)

    result = {}
    for dest in dests:
        if orgn not in G or dest not in G:
            result[(orgn, dest)] = []
            continue
        try:
            paths = list(islice(
                nx.shortest_simple_paths(G, orgn, dest, weight='weight'),
                k
            ))
            result[(orgn, dest)] = paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            result[(orgn, dest)] = []

    return result


def generate_path_cache(network, networkName, k, first_thru_node,
                        output_dir="Paths", num_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    safe_name = networkName.replace(" ", "_")
    file_path = os.path.join(output_dir, f'allPathsODCache_{k}_{safe_name}.pkl')

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    edges = [(u, v, link.fft) for (u, v), link in network.linkSet.items()]

    origin_tasks = defaultdict(list)
    for (orgn, dest), trip in network.tripSet.items():
        if trip.demand > 0 and orgn != dest:
            origin_tasks[int(orgn)].append(int(dest))

    tasks = [(orgn, dests, k) for orgn, dests in origin_tasks.items()]

    print(f"Launching {len(tasks)} origin tasks across {num_workers} workers...")

    allPathOD = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(edges, first_thru_node),
    ) as executor:
        futures = [executor.submit(process_origin, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Computing OD paths"):
            allPathOD.update(future.result())

    # Sanity check, count violating paths not violating hops
    violations = 0
    for (o, d), paths in allPathOD.items():
        for path in paths:
            if any(int(n) < first_thru_node for n in path[1:-1]):
                violations += 1

    if violations == 0:
        print("Verification passed: no centroid used as thru node.")
    else:
        print(f"WARNING: {violations} violating paths found.")
        raise ValueError("Centroid used as thru node. Check path generation logic.")

    with open(file_path, 'wb') as f:
        pickle.dump(allPathOD, f)
    print(f"Saved to {file_path}")

    # Metadata
    num_nodes = len(network.nodeSet)
    num_thru_nodes = len(network.nodeSet) - (first_thru_node - 1) if first_thru_node > 1 else len(network.nodeSet)
    num_links = len(network.linkSet)
    num_od_pairs = len(allPathOD)

    # Zone count from trip table, more reliable than first_thru_node
    zones = set()
    for (o, d) in network.tripSet.keys():
        zones.add(int(o))
        zones.add(int(d))
    num_zones = len(zones)

    path_counts = [len(paths) for paths in allPathOD.values()]
    mean_paths_per_od = sum(path_counts) / num_od_pairs if num_od_pairs > 0 else 0
    max_paths_per_od = max(path_counts) if path_counts else 0

    return (file_path, num_nodes, num_thru_nodes, num_links,
            num_od_pairs, num_zones, mean_paths_per_od, max_paths_per_od)

    
if __name__ == "__main__":

    first_thru_node_dict = {
        # "Sioux Falls": 1,
        # "BMC": 37,
        "Winnipeg": 148,
        # "EMA": 1,
        # "Anaheim": 39,
        # "Berlin Center": 866,
        # "Chicago Sketch": 1,
        # "Birmingham": 899,
        # "Philadelphia": 1526,
        # "Chicago Regional": 1791,
        # "Gold Coast": 1069,
    }

    num_paths_dict = {name: 20 for name in first_thru_node_dict}
    num_paths_dict["Chicago Regional"] = 5
    num_paths_dict["Philadelphia"] = 5

    metadata_file = "path_cache_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("Network,NumNodes,NumThruNodes,NumLinks,NumODPairs,NumZones,MeanPathsPerOD,MaxPathsPerOD, Type\n")

    for networkName in first_thru_node_dict:
        k = num_paths_dict[networkName]
        safe_name = networkName.replace(" ", "_")
        cache_path = os.path.join("Paths", f'allPathsODCache_{k}_{safe_name}.pkl')
        # if os.path.exists(cache_path):
        #     print(f"Skipping {networkName}, cache already exists at {cache_path}")
        #     continue

        inputLocation = f'testNetworks/{networkName}/'
        convert_network_tntp_to_dat(f'{inputLocation}/net.tntp',
                                    f'{inputLocation}/network.dat')
        convert_demand_tntp_to_dat(f'{inputLocation}/trips.tntp',
                                    f'{inputLocation}/demand.dat')

        network = Network()
        network.readNetwork(inputLocation)
        network.readDemand(inputLocation)

        result = generate_path_cache(
            network=network,
            networkName=networkName,
            k=k,
            first_thru_node=first_thru_node_dict[networkName],
            output_dir="Paths",
        )
        (file_path, num_nodes, num_thru_nodes, num_links,
         num_od_pairs, num_zones, mean_paths_per_od, max_paths_per_od) = result

        with open(metadata_file, "a") as f:
            f.write(f"{networkName},{num_nodes},{num_thru_nodes},{num_links},"
                    f"{num_od_pairs},{num_zones},{mean_paths_per_od:.2f},"
                    f"{max_paths_per_od}, Yen\n")