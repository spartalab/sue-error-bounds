"""
capacityDesign.py
Capacity expansion example of Section subsec:4_netdes (Algorithm alg:4_ls) on Chicago Sketch, run once
without and once with the TSTT bounds.

A design is a set of links whose capacity is doubled. Links with fft = 0 (centroid connectors) are not
candidates, which leaves 2,176 candidate links. A design is feasible if the total length of its links
is at most BUDGET miles.

Objective (minimized), in dollars:
    Z(design) = ALPHA * TSTT(design) + C(design)
    C(design) = COST_PER_LANE_MILE / LANE_CAPACITY * sum over the design of length * capacity
Z(no upgrade) - Z(design) is the net benefit of the design.

Local search (the same in both runs, all random choices from SEED):
    1. Start from a random design within the budget, solved to RGAP FINAL_RGAP.
    2. Try the neighbours of the current design (add, drop or swap one link) in random order. The first
       neighbour with Z < Z(current) becomes the current design.
    3. Stop when no neighbour is accepted (local minimum) or when the time limit is reached.
Without bounds, every neighbour is solved to FINAL_RGAP. With bounds, the assignment of a neighbour is
run (and resumed) to each RGAP in CHECK_RGAPS, and the neighbour is pruned as soon as the lower bound on
Z from Corollary cor:3_tsttconvex reaches Z(current). Otherwise it is solved to FINAL_RGAP. Both runs
make the same moves in the same order.

A design whose Z (or a pruning bound) is already known is not solved again, but still counts as an
evaluation. Results (one row per evaluation) are written to OUTPUT_FOLDER, and a run whose CSV file
already exists is not run again. A summary with every number reported in Section subsec:4_netdes is
printed and saved as a text file in OUTPUT_FOLDER, and the two figures of the section are saved to PAPER_FOLDER.
"""
import os
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from readNetwork import Network
from SUEPath import StochasticUE
from helper import convert_network_tntp_to_dat, convert_demand_tntp_to_dat


NETWORK = "Chicago Sketch"  # folder name in testNetworks/, as in main.py
BUDGET = 40.0               # B: largest total length of the upgraded links (miles)
TIME_LIMIT_HOURS = 1.0      # time limit of each run
SEED = 42                   # start design and the order in which neighbours are tried

# Objective
ALPHA = 2882.0              # $ per vehicle-minute: present value over 30 years at 4% of $20 per vehicle-hour for 500 peak hours per year
COST_PER_LANE_MILE = 6e6    # $ per lane-mile
LANE_CAPACITY = 1900        # vehicles per hour per lane

THETA = 0.5
STEP_RULE = "BB1"
K = 20                      # paths per OD pair (Yen's k-shortest paths), as in main.py
# Passed to every assignment call, as in main.py
OPTIONS = {"stepRule": STEP_RULE, "initType": "logitFreeFlow", "I_s": 10,
           "demand_multiplier": 1.0, "time_limit": 60*60, "save_log": False}
CHECK_RGAPS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]    # RGAPs where the bound is checked (run with bounds)
FINAL_RGAP = 1e-12                              # RGAP of the exact TSTT, used by both runs
MAX_ITER = 100000
OUTPUT_FOLDER = "capDesRes/"
PAPER_FOLDER = "paperResults/"

# Figures
ORANGE = "#eb6834"     # without bounds
BLUE = "#2a78d6"       # with bounds
PRUNED = "#6aa6ea"     # pruned designs
ACCEPTED = "black"
# Both figures have the same size and the same plot area, so they can sit side by side (each 0.5\textwidth)
FIG_SIZE = (3.25, 3.5)
MARGINS = {"left": 0.2, "right": 0.82, "top": 0.97, "bottom": 0.34}
LEGEND = {"frameon": False, "ncol": 2, "handlelength": 1.5, "handletextpad": 0.4, "columnspacing": 0.8}


def result_file(bounds):
    """CSV file of the run without (bounds=False) or with (bounds=True) bounds."""
    return (f"{OUTPUT_FOLDER}LS_results_{bounds}_{NETWORK.replace(' ', '_')}_B{BUDGET:g}"
            f"_T{TIME_LIMIT_HOURS:g}h_S{SEED}_A{ALPHA:g}_{STEP_RULE}.csv")


def set_design(network, base_capacity, design):
    """Restore every capacity, then double the capacity of each link in the design."""
    for link, capacity in base_capacity.items():
        network.linkSet[link].capacity = capacity
    for link in design:
        network.linkSet[link].capacity = 2 * base_capacity[link]


def solve(sue, accuracy, resume):
    """
    resume=False: start a new assignment from the free-flow logit flows.
    resume=True:  continue the previous assignment from its current h, BB1 history
                  and iteration count, exactly as if it had never stopped.
    """
    sue.assignment(accuracy=accuracy, maxIter=MAX_ITER, networkName=NETWORK, resume=resume, **OPTIONS)
    if not sue.gap < accuracy:
        raise RuntimeError(f"Assignment stopped at RGAP {sue.gap} without reaching {accuracy}")


def construction_cost(network, base_capacity, design):
    """C: COST_PER_LANE_MILE / LANE_CAPACITY dollars per mile for each unit of added capacity."""
    return sum(COST_PER_LANE_MILE / LANE_CAPACITY * network.linkSet[link].length * base_capacity[link]
               for link in design)


def design_length(network, design):
    return sum(network.linkSet[link].length for link in design)


def design_name(design):
    return str(sorted(f"{link[0]}-{link[1]}" for link in design))


def random_start(network, candidates, rng):
    """Shuffled candidates, added while the total length stays below a target drawn from [0, BUDGET]."""
    target = rng.uniform(0, BUDGET)
    design, used = [], 0.0
    for i in rng.permutation(len(candidates)):
        length = network.linkSet[candidates[i]].length
        if used + length <= target:
            design.append(candidates[i])
            used += length
    return frozenset(design)


def neighbours(network, candidates, design, rng):
    """Every design one add, drop or swap away that fits in BUDGET, in random order, with the move made."""
    used = design_length(network, design)
    inside = [link for link in candidates if link in design]
    outside = [link for link in candidates if link not in design]
    found = []
    for a in outside:
        if used + network.linkSet[a].length <= BUDGET:
            found.append((design | {a}, f"add {a[0]}-{a[1]}"))
    for d in inside:
        found.append((design - {d}, f"drop {d[0]}-{d[1]}"))
    for d in inside:
        for a in outside:
            if used - network.linkSet[d].length + network.linkSet[a].length <= BUDGET:
                found.append(((design - {d}) | {a}, f"swap {d[0]}-{d[1]} for {a[0]}-{a[1]}"))
    return [found[i] for i in rng.permutation(len(found))]


def tstt_lower_bound(sue):
    """Lower bound on the equilibrium TSTT at the solver's current h (Corollary cor:3_tsttconvex)."""
    return float(sue.evaluateBounds()["tsttLowerConvex"])


def local_search(network, sue, base_capacity, candidates, z_base, bounds):
    """Algorithm alg:4_ls. bounds=False: every design is solved to FINAL_RGAP."""
    print("Network:", NETWORK, "| Bounds:", bounds)
    rng = np.random.default_rng(SEED)
    exact = {}          # design -> exact TSTT
    lower = {}          # design -> largest lower bound on its TSTT found so far
    rows = []
    start = time.time()

    def out_of_time():
        return time.time() - start >= TIME_LIMIT_HOURS * 3600

    def evaluate(design, threshold, move, step):
        """
        Z of design, or None when the bound shows Z >= threshold (pruned). threshold=None (start design)
        means always solve to FINAL_RGAP. Adds one row per call, also for designs already known.
        """
        construction = construction_cost(network, base_capacity, design)
        row = {
            'Step': step,
            'Move': move,
            'Design': design_name(design),
            'Upgraded length': design_length(network, design),
            'Construction cost': construction,
            'Threshold Z': threshold,
            'Cached': True,
            'Stopped at RGAP': None,
        }
        if design in exact:
            z = ALPHA * exact[design] + construction
        elif bounds and threshold is not None and design in lower and ALPHA * lower[design] + construction >= threshold:
            z = None
        else:
            row['Cached'] = False
            set_design(network, base_capacity, design)
            rgaps = CHECK_RGAPS + [FINAL_RGAP] if bounds and threshold is not None else [FINAL_RGAP]
            for i, rgap in enumerate(rgaps):
                solve(sue, rgap, resume=i > 0)
                if rgap == FINAL_RGAP:
                    break
                lb = tstt_lower_bound(sue)
                lower[design] = max(lower.get(design, -np.inf), lb)
                if ALPHA * lb + construction >= threshold:
                    break
            row['Stopped at RGAP'] = rgap
            if rgap == FINAL_RGAP:
                exact[design] = float(sue.tstt)
                z = ALPHA * exact[design] + construction
            else:
                z = None

        row.update({
            'LB TSTT': lower.get(design),
            'Exact TSTT': exact.get(design),
            'Z': z,
            'Pruned': z is None,
            'Accepted': threshold is None or (z is not None and z < threshold),
            'Time spent': time.time() - start,
        })
        rows.append(row)
        print("Evaluation:", len(rows), "|", move, "| Stopped at RGAP:", row['Stopped at RGAP'],
              "| Accepted:", row['Accepted'], "| Time:", round(row['Time spent']), "seconds")
        return z

    current = random_start(network, candidates, rng)
    z_current = evaluate(current, None, "start", 0)
    step = 0
    improved = True
    while improved and not out_of_time():
        improved = False
        step += 1
        for design, move in neighbours(network, candidates, current, rng):
            if out_of_time():
                break
            threshold = z_current
            z = evaluate(design, threshold, move, step)
            if z is not None and z < threshold:
                current, z_current, improved = design, z, True
                break

    df = pd.DataFrame(rows)
    df['Best Z'] = df['Z'].cummin()
    df['Net benefit of best'] = z_base - df['Best Z']
    df.to_csv(result_file(bounds), index=False)
    return df


def summary_file():
    """Text file with the numbers reported in Section subsec:4_netdes."""
    return (f"{OUTPUT_FOLDER}LS_summary_{NETWORK.replace(' ', '_')}_B{BUDGET:g}"
            f"_T{TIME_LIMIT_HOURS:g}h_S{SEED}_A{ALPHA:g}_{STEP_RULE}.txt")


def write_summary(without, bounded, z_base):
    """The numbers reported in Section subsec:4_netdes, printed and saved to summary_file()."""
    K = len(without)
    same = bounded.iloc[:K]
    pruned = bounded[bounded['Pruned']]
    solved = bounded[~bounded['Pruned'] & ~bounded['Cached'] & (bounded['Step'] > 0)]
    timeWithout = without['Time spent'].iloc[-1]
    timeWith = same['Time spent'].iloc[-1]
    benefitWithout = z_base - without['Z'].min()
    benefitWith = z_base - bounded['Z'].min()

    # Seconds of each evaluation ('Time spent' is cumulative)
    secondsWithout = without['Time spent'].diff().fillna(without['Time spent'].iloc[0])
    secondsWith = bounded['Time spent'].diff().fillna(bounded['Time spent'].iloc[0])

    # Time to reach CHECK_RGAPS[0] as a share of a full solve: mean time of a design pruned at CHECK_RGAPS[0]
    # (run with bounds) over the mean time of a neighbour solved to FINAL_RGAP (run without bounds)
    prunedFirst = bounded['Pruned'] & ~bounded['Cached'] & (bounded['Stopped at RGAP'] == CHECK_RGAPS[0])
    fullSolve = ~without['Cached'] & (without['Step'] > 0)
    share = secondsWith[prunedFirst].mean() / secondsWithout[fullSolve].mean()

    # Best case for the first K evaluations: every rejected neighbour of the run without bounds
    # had been pruned at CHECK_RGAPS[0], i.e. had taken share times its full solve time
    rejected = fullSolve & ~without['Accepted']
    bestCase = secondsWithout[~rejected].sum() + share * secondsWithout[rejected].sum()

    prunedAt = {rgap: int((pruned['Stopped at RGAP'] == rgap).sum()) for rgap in CHECK_RGAPS}
    prunedFromCache = int(pruned['Cached'].sum())

    lines = [
        f"Network: {NETWORK}, theta = {THETA}, budget = {BUDGET:g} miles, time limit = {TIME_LIMIT_HOURS:g} h, "
        f"seed = {SEED}, step rule = {STEP_RULE}",
        "",
        f"Evaluations without bounds: {K} in {timeWithout:.0f} s",
        f"Evaluations with bounds: {len(bounded)} in {bounded['Time spent'].iloc[-1]:.0f} s",
        f"Pruned with bounds: {len(pruned)} of {len(bounded)} ({100 * len(pruned) / len(bounded):.0f}%)",
    ]
    for rgap in CHECK_RGAPS:
        lines.append(f"    pruned at RGAP {rgap:g}: {prunedAt[rgap]}")
    lines += [
        f"    pruned again from a cached bound: {prunedFromCache}",
        f"Pruned at an RGAP tighter than {CHECK_RGAPS[0]:g}: {len(pruned) - prunedAt[CHECK_RGAPS[0]] - prunedFromCache}",
        f"Neighbours solved to RGAP {FINAL_RGAP:g} with bounds: {len(solved)}, of which accepted: {int(solved['Accepted'].sum())}",
        f"Same moves in the first {K} evaluations: {same['Design'].tolist() == without['Design'].tolist()}",
        "",
        f"Time of the first {K} evaluations: {timeWith:.0f} s with bounds, {timeWithout:.0f} s without "
        f"({100 * (1 - timeWith / timeWithout):.0f}% less)",
        f"Net benefit of the best design after {TIME_LIMIT_HOURS:g} h: ${benefitWith / 1e6:,.1f}M with bounds, "
        f"${benefitWithout / 1e6:,.1f}M without",
        "",
        f"Time to reach RGAP {CHECK_RGAPS[0]:g}, as a share of a full solve: {100 * share:.0f}%",
        f"    (mean time of a design pruned at {CHECK_RGAPS[0]:g} with bounds: {secondsWith[prunedFirst].mean():.2f} s,",
        f"     mean time of a neighbour solved to {FINAL_RGAP:g} without bounds: {secondsWithout[fullSolve].mean():.2f} s)",
        f"First {K} evaluations if every rejected neighbour had been pruned at {CHECK_RGAPS[0]:g}: {bestCase:.0f} s "
        f"({100 * (1 - bestCase / timeWithout):.0f}% less than without bounds)",
    ]
    print("=" * 70)
    print("\n".join(lines))
    print("=" * 70)
    with open(summary_file(), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_results(without, bounded):
    """Figure 1: progress over time. Figure 2: designs evaluated with bounds, by the RGAP where they stopped."""
    # Same fonts as genPaperPlots.py: LaTeX if it is installed, otherwise matplotlib's LaTeX-like math text
    if shutil.which("latex"):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": r"\usepackage{amsmath,amssymb}"})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"})
    plt.rcParams.update({"font.size": 9})

    # Figure 1: best net benefit (left axis) and number of designs evaluated (right axis) against time
    fig, left = plt.subplots(figsize=FIG_SIZE)
    right = left.twinx()

    # Time saved: when each run found the best design of the run without bounds
    target = without["Net benefit of best"].dropna().iloc[-1]
    tWith = bounded[bounded["Net benefit of best"] >= target - 1]["Time spent"].iloc[0] / 60
    tWithout = without[without["Net benefit of best"] >= target - 1]["Time spent"].iloc[0] / 60
    saved = right.axvspan(tWith, tWithout, color="0.88", linewidth=0, zorder=0, label="Time saved")

    handles = [saved]
    # Line style as well as color, so a black and white print can tell all four lines apart
    for run, label, color, bestStyle, countStyle in [(without, "no bounds", ORANGE, "-.", ":"),
                                                      (bounded, "bounds", BLUE, "-", "--")]:
        minutes = run["Time spent"] / 60
        best = run["Net benefit of best"].ffill() / 1e6
        line, = left.step(minutes, best, where="post", color=color, linewidth=2, linestyle=bestStyle,
                          label=f"Benefit, {label}")
        count, = right.step(minutes, np.arange(1, len(run) + 1), where="post", color=color, linewidth=1,
                            linestyle=countStyle, label=f"Designs, {label}")
        handles += [line, count]

    left.axhline(0, color="black", linewidth=0.8)
    # Best design of each run at the time limit (named in the legend: there is no room for text at half page width)
    for run in [without, bounded]:
        best = left.axhline(run["Net benefit of best"].dropna().iloc[-1] / 1e6, color="0.4", linewidth=0.8,
                            linestyle=":", zorder=1, label=f"Best after {TIME_LIMIT_HOURS:g} h")
    handles.append(best)
    left.set_ylabel(r"Best net benefit (\$M)")
    right.set_ylabel("Designs evaluated")
    left.set_xlabel("Time (min)")
    left.set_xlim(0, TIME_LIMIT_HOURS * 60)
    left.set_zorder(right.get_zorder() + 1)
    left.patch.set_visible(False)
    fig.subplots_adjust(**MARGINS)
    fig.legend(handles=[handles[i] for i in (1, 3, 5, 2, 4, 0)], loc="lower center", **LEGEND,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(PAPER_FOLDER + "LS_progress_L40_C2950.pdf")
    plt.close(fig)

    # Figure 2: every design evaluated by the run with bounds: one column per RGAP where it was stopped,
    # placed by how much it changed Z from the last accepted design (for a pruned design, a lower bound on the increase)
    change = np.where(bounded["Pruned"], ALPHA * bounded["LB TSTT"] + bounded["Construction cost"], bounded["Z"]) - bounded["Threshold Z"]
    change = change.values / 1e6     # $M, negative = Z improved
    pruned = bounded["Pruned"].values
    stopped = bounded["Stopped at RGAP"].values
    # One column per RGAP in CHECK_RGAPS (pruned designs), then FINAL_RGAP (designs solved exactly)
    ROWS = []
    for rgap in CHECK_RGAPS:
        ROWS.append((pruned & (stopped == rgap), PRUNED, "o", rgap))
    ROWS.append(((~pruned) & (bounded["Step"].values > 0), ACCEPTED, "v", FINAL_RGAP))

    fig, axes = plt.subplots(figsize=FIG_SIZE)
    labels = []
    for position, (rows, color, marker, rgap) in enumerate(ROWS):
        axes.plot(np.full(rows.sum(), position), change[rows], marker, color=color, markersize=6, alpha=0.5,
                  markeredgecolor="black", markeredgewidth=0.3, linestyle="")
        # RGAP, number of designs, and their share of all evaluations
        labels.append(f"$10^{{{round(np.log10(rgap))}}}$\n({rows.sum()})\n${100 * rows.sum() / len(bounded):.0f}\\%$")
    axes.set_xticks(range(len(ROWS)))
    axes.set_xlim(-0.5, len(ROWS) - 0.5)
    axes.set_xticklabels(labels, fontsize=8)     # 8 pt so the columns of labels do not touch
    axes.set_xlabel("Stopped at RGAP (number and share of designs)")
    axes.set_yscale("symlog", linthresh=1)
    axes.set_yticks([-100, -10, -1, 0, 1, 10, 100], ["$-100$", "$-10$", "$-1$", "$0$", "$1$", "$10$", "$100$"])
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_ylim(-200, 200)
    axes.set_ylabel(r"Change in Z from last design (\$M)")
    axes.grid(axis="y", color="0.9", linewidth=0.6)
    axes.set_axisbelow(True)
    fig.subplots_adjust(**MARGINS)
    prunedHandle = Line2D([], [], marker="o", linestyle="", color=PRUNED, markeredgecolor="black",
                          markeredgewidth=0.3, label="Pruned (lower bound)")
    acceptedHandle = Line2D([], [], marker="v", linestyle="", color="0.5", markeredgecolor="black",
                            markeredgewidth=0.3, label="Accepted (exact)")
    fig.legend(handles=[prunedHandle, acceptedHandle], loc="upper center", **LEGEND, bbox_to_anchor=(0.5, 0.12))
    fig.savefig(PAPER_FOLDER + "LS_evaluations_L40_C2950.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(PAPER_FOLDER, exist_ok=True)

    folder = "testNetworks/" + NETWORK + "/"
    if not os.path.exists(folder + "network.dat"):
        convert_network_tntp_to_dat(folder + "net.tntp", folder + "network.dat")
    if not os.path.exists(folder + "demand.dat"):
        convert_demand_tntp_to_dat(folder + "trips.tntp", folder + "demand.dat")
    network = Network()
    network.readNetwork(folder)
    network.readDemand(folder)
    base_capacity = {link: network.linkSet[link].capacity for link in network.linkSet}
    candidates = [link for link in network.linkSet if network.linkSet[link].fft > 0]

    # The path set, D, W_l and S_l do not depend on capacities, and assignment(resume=False) resets h,
    # the BB1 history and the iteration count. SUEPath reads capacities from network.linkSet.
    sue = StochasticUE(network, NETWORK, THETA, False, K)

    solve(sue, FINAL_RGAP, resume=False)     # no upgrade
    z_base = ALPHA * float(sue.tstt)
    print("Candidates:", len(candidates), f"| No-upgrade Z ${z_base / 1e6:,.3f}M")

    runs = {}
    for bounds in [False, True]:
        if os.path.exists(result_file(bounds)):
            runs[bounds] = pd.read_csv(result_file(bounds))
        else:
            runs[bounds] = local_search(network, sue, base_capacity, candidates, z_base, bounds)
    write_summary(runs[False], runs[True], z_base)
    plot_results(runs[False], runs[True])
    set_design(network, base_capacity, ())