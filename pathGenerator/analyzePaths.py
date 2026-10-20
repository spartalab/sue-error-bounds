"""
Analyze diversity/overlap of generated path sets for all networks.
Saves:
  - path_diversity_summary.csv    (cross-network summary)
  - path_diversity_per_od.csv     (per-OD metrics for all networks)
  - path_diversity_log.txt        (everything printed to console)
All output files are written next to this script.
"""

import os
import sys
import csv
import pickle
import statistics
from collections import defaultdict
from itertools import combinations

# helper.py and readNetwork.py live in the parent folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from helper import convert_network_tntp_to_dat, convert_demand_tntp_to_dat
from readNetwork import Network


LIST_OF_NETWORKS = [
    "testNetworks/Sioux Falls/",
    # "testNetworks/BMC/",
    # "testNetworks/EMA/",
    "testNetworks/Anaheim/",
    "testNetworks/Chicago Sketch/",
    # "testNetworks/Berlin Center/",
    "testNetworks/Winnipeg/",
    # "testNetworks/Philadelphia/",
    # "testNetworks/Chicago Regional/",
]

LIST_OF_PICKLE_PATHS = [
    "Paths/allPathsODCache_20_Sioux_Falls.pkl",
    # "Paths/allPathsODCache_20_BMC.pkl",
    # "Paths/allPathsODCache_20_EMA.pkl",
    "Paths/allPathsODCache_20_Anaheim.pkl",
    "Paths/allPathsODCache_20_Chicago_Sketch.pkl",
    # "Paths/allPathsODCache_20_Berlin_Center.pkl",
    "Paths/allPathsODCache_20_Winnipeg.pkl",
    # "Paths/allPathsODCache_5_Philadelphia.pkl",
    # "Paths/allPathsODCache_5_Chicago_Regional.pkl",
]

LIST_OF_NETWORKS_NAMES = [net.split("/")[1] for net in LIST_OF_NETWORKS]

# Resolve relative paths against parent dir so we can run from anywhere
LIST_OF_NETWORKS = [
    p if os.path.isabs(p) else os.path.join(PARENT_DIR, p)
    for p in LIST_OF_NETWORKS
]
LIST_OF_PICKLE_PATHS = [
    p if os.path.isabs(p) else os.path.join(PARENT_DIR, p)
    for p in LIST_OF_PICKLE_PATHS
]

CSV_SUMMARY = os.path.join(SCRIPT_DIR, "path_diversity_summary.csv")
CSV_PER_OD = os.path.join(SCRIPT_DIR, "path_diversity_per_od.csv")
LOG_FILE   = os.path.join(SCRIPT_DIR, "path_diversity_log.txt")


# Tee prints to both console and log file
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def path_edges(path):
    return set(zip(path[:-1], path[1:]))


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def overlap_min(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def path_cost(path, fft):
    return sum(fft[(path[i], path[i+1])] for i in range(len(path) - 1))


def summarize(values, name):
    if not values:
        print(f"  {name}: (no data)")
        return
    values = sorted(values)
    n = len(values)
    print(f"  {name}")
    print(f"    mean={statistics.mean(values):.3f}  "
          f"median={statistics.median(values):.3f}  "
          f"p10={values[int(0.10*n)]:.3f}  "
          f"p90={values[int(0.90*n)]:.3f}  "
          f"min={values[0]:.3f}  max={values[-1]:.3f}")


def analyze_network(network_name, input_location, pickle_path, per_od_writer):
    print("=" * 70)
    print(f"NETWORK: {network_name}")
    print("=" * 70)

    if not os.path.exists(pickle_path):
        print(f"  Pickle not found: {pickle_path}  (skipping)\n")
        return None
    if not os.path.isdir(input_location):
        print(f"  Network folder not found: {input_location}  (skipping)\n")
        return None

    convert_network_tntp_to_dat(f'{input_location}/net.tntp',
                                f'{input_location}/network.dat')
    convert_demand_tntp_to_dat(f'{input_location}/trips.tntp',
                               f'{input_location}/demand.dat')
    network = Network()
    network.readNetwork(input_location)
    network.readDemand(input_location)
    fft = {(int(u), int(v)): link.fft
           for (u, v), link in network.linkSet.items()}

    with open(pickle_path, "rb") as f:
        all_paths = pickle.load(f)

    print(f"Loaded {len(all_paths)} OD pairs from {pickle_path}\n")

    count_dist = defaultdict(int)
    for p in all_paths.values():
        count_dist[len(p)] += 1
    print("Paths per OD distribution:")
    for c in sorted(count_dist):
        print(f"  {c:>3} paths : {count_dist[c]} OD pairs")
    print()

    mean_edge_j = []
    max_edge_j = []
    mean_overlap = []
    max_overlap = []
    len_ratios = []
    cost_sd = []
    cost_cv = []
    cost_range_ratio = []

    for od, paths in all_paths.items():
        if len(paths) < 2:
            continue
        edge_sets = [path_edges(p) for p in paths]
        lengths = [len(p) - 1 for p in paths]
        try:
            costs = [path_cost([int(n) for n in p], fft) for p in paths]
        except KeyError as e:
            if not hasattr(analyze_network, "_warned"):
                print(f"  WARNING: fft lookup failed for edge {e}. "
                    f"Sample fft key: {next(iter(fft))}, "
                    f"sample path edge: {(paths[0][0], paths[0][1])}")
                analyze_network._warned = True
            raise e
        ejs, ovs = [], []
        for i, j in combinations(range(len(paths)), 2):
            ejs.append(jaccard(edge_sets[i], edge_sets[j]))
            ovs.append(overlap_min(edge_sets[i], edge_sets[j]))

        m_ej = statistics.mean(ejs)
        x_ej = max(ejs)
        m_ov = statistics.mean(ovs)
        x_ov = max(ovs)
        lr = max(lengths) / min(lengths) if min(lengths) > 0 else 0.0

        mean_edge_j.append(m_ej)
        max_edge_j.append(x_ej)
        mean_overlap.append(m_ov)
        max_overlap.append(x_ov)
        if min(lengths) > 0:
            len_ratios.append(lr)

        sd = mu = cv = rr = 0.0
        if costs is not None and len(costs) > 1:
            sd = statistics.stdev(costs)
            mu = statistics.mean(costs)
            cost_sd.append(sd)
            if mu > 0:
                cv = sd / mu
                cost_cv.append(cv)
            if min(costs) > 0:
                rr = (max(costs) - min(costs)) / min(costs)
                cost_range_ratio.append(rr)

        o, d = od
        per_od_writer.writerow([
            network_name, o, d, len(paths),
            f"{m_ej:.6f}", f"{x_ej:.6f}",
            f"{m_ov:.6f}", f"{x_ov:.6f}",
            min(lengths), max(lengths), f"{lr:.6f}",
            f"{sd:.6f}", f"{mu:.6f}", f"{cv:.6f}", f"{rr:.6f}",
        ])

    print(f"OD pairs with >= 2 paths: {len(mean_edge_j)}")
    print("Overlap interpretation: 0 = disjoint, 1 = identical\n")

    summarize(mean_edge_j, "Mean pairwise EDGE Jaccard (per OD)")
    summarize(max_edge_j,  "Max  pairwise EDGE Jaccard (per OD)")
    summarize(mean_overlap, "Mean edge overlap ratio shared/min (per OD)")
    summarize(max_overlap,  "Max  edge overlap ratio shared/min (per OD)")
    summarize(len_ratios,   "Length ratio (longest/shortest, in edges)")
    print()
    summarize(cost_sd,          "Free-flow cost SD (per OD)")
    summarize(cost_cv,          "Free-flow cost CV = SD/mean (per OD)")
    summarize(cost_range_ratio, "Cost range ratio (max-min)/min (per OD)")

    threshold = 0.8
    flagged = sum(1 for v in max_overlap if v >= threshold)
    if max_overlap:
        print(f"\nOD pairs with some path-pair overlap >= {threshold}: "
              f"{flagged} / {len(max_overlap)} "
              f"({100 * flagged / len(max_overlap):.1f}%)")
    print()

    return {
        "network": network_name,
        "num_od": len(all_paths),
        "num_od_multi": len(mean_edge_j),
        "mean_overlap": statistics.mean(mean_overlap) if mean_overlap else 0,
        "mean_edge_jaccard": statistics.mean(mean_edge_j) if mean_edge_j else 0,
        "mean_cost_sd": statistics.mean(cost_sd) if cost_sd else 0,
        "mean_cost_cv": statistics.mean(cost_cv) if cost_cv else 0,
        "mean_cost_range_ratio": statistics.mean(cost_range_ratio) if cost_range_ratio else 0,
        "pct_flagged": (100 * flagged / len(max_overlap)) if max_overlap else 0,
    }


# Open log + per-OD CSV and run
original_stdout = sys.stdout
log_f = open(LOG_FILE, "w")
sys.stdout = Tee(original_stdout, log_f)

per_od_f = open(CSV_PER_OD, "w", newline="")
per_od_writer = csv.writer(per_od_f)
per_od_writer.writerow([
    "network", "origin", "dest", "num_paths",
    "mean_edge_jaccard", "max_edge_jaccard",
    "mean_edge_overlap_min", "max_edge_overlap_min",
    "min_len_edges", "max_len_edges", "length_ratio",
    "cost_sd", "cost_mean", "cost_cv", "cost_range_ratio",
])

all_summaries = []
try:
    for name, folder, pkl in zip(LIST_OF_NETWORKS_NAMES,
                                 LIST_OF_NETWORKS,
                                 LIST_OF_PICKLE_PATHS):
        s = analyze_network(name, folder, pkl, per_od_writer)
        if s is not None:
            all_summaries.append(s)

    if all_summaries:
        print("=" * 70)
        print("CROSS-NETWORK SUMMARY")
        print("=" * 70)
        print(f"{'Network':<20}{'ODs':>8}{'meanOv':>10}{'meanJac':>10}"
              f"{'costCV':>10}{'costRR':>10}{'%flag':>8}")
        for s in all_summaries:
            print(f"{s['network']:<20}{s['num_od']:>8}"
                  f"{s['mean_overlap']:>10.3f}{s['mean_edge_jaccard']:>10.3f}"
                  f"{s['mean_cost_cv']:>10.3f}{s['mean_cost_range_ratio']:>10.3f}"
                  f"{s['pct_flagged']:>8.1f}")

    with open(CSV_SUMMARY, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "num_od", "num_od_multi",
                    "mean_overlap", "mean_edge_jaccard",
                    "mean_cost_sd", "mean_cost_cv",
                    "mean_cost_range_ratio", "pct_flagged_overlap_ge_0.8"])
        for s in all_summaries:
            w.writerow([s["network"], s["num_od"], s["num_od_multi"],
                        f"{s['mean_overlap']:.6f}",
                        f"{s['mean_edge_jaccard']:.6f}",
                        f"{s['mean_cost_sd']:.6f}",
                        f"{s['mean_cost_cv']:.6f}",
                        f"{s['mean_cost_range_ratio']:.6f}",
                        f"{s['pct_flagged']:.3f}"])

    print()
    print(f"Saved: {CSV_SUMMARY}")
    print(f"Saved: {CSV_PER_OD}")
    print(f"Saved: {LOG_FILE}")
finally:
    per_od_f.close()
    sys.stdout = original_stdout
    log_f.close()
