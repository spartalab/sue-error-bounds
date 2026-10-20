"""
genPaperPlots.py

Writes table.tex and the four figures of Section sec:results from the bound runs
that SUEPath writes to boundResults/<Network>/theta<theta>_d<d>x_K<k>_<stepRule>/plots.csv.
All settings are hardcoded below.  main.py calls makePaperPlots after solving; to redo only
the tables and figures from saved runs, run:  python genPaperPlots.py

Output, in OUTPUT_FOLDER:
    table.tex                                   one block per table; paste each between \\midrule and \\bottomrule
    objective_sandwich_theta_<theta>.pdf        Anaheim | Winnipeg | Chicago Sketch
    bounds_ChicagoSketch_theta_<theta>.pdf      Link flow | Link cost | TSTT
    tstt_value_theta_<theta>.pdf                Anaheim | Winnipeg | Chicago Sketch
    convergence_ChicagoSketch_theta_<theta>.pdf Objective | Link flow | TSTT: bound and ||h - hat h||(^2) vs iteration
"""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # draw to files only
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullLocator


# =============================================================================
# SETTINGS
# =============================================================================

# Networks in table order: folder name in boundResults/ -> k (paths per OD pair)
NETWORKS = {
    "Sioux Falls": 20,
    "Anaheim": 20,
    "Winnipeg": 20,
    "Chicago Sketch": 20,
}
# Name shown in the tables and figure titles, where it differs from the folder name
DISPLAY_NAMES = {
}
THETAS = [0.5, 1.5]                     # one group of table columns each
STEP_RULE = "BB1"
DEMAND_MULTIPLIER = 1.0
BOUND_ACCURACY = 1e-8                   # every run is cut at the first iteration with RGAP <= this
TABLE_RGAP_LEVELS = [1e-4, 1e-6, 1e-8]  # rows of the tables

COST_NETWORKS = ["Anaheim", "Winnipeg", "Chicago Sketch"]   # columns of tab:4_cost
COST_THETA = 0.5

FIGURE_NETWORKS = ["Sioux Falls", "Winnipeg", "Chicago Sketch"] # panels of figures 1 and 4
BOUND_FIGURE_NETWORKS = ["Chicago Sketch"]      # one figure each (figures 2 and 3)
FIGURE_THETA = 0.5

# Convergence of the bounds against ||h - hat h|| (Theorem thm:boundconvergence)
CONVERGENCE_NETWORKS = ["Chicago Sketch"]       # one figure each: objective | link flow | TSTT
CONVERGENCE_TABLE_THETA = 0.5                   # theta of tab:4_convergence
CONVERGENCE_START_RGAP = 1e-4                   # convergence figures and tab:4_convergence use RGAP <= this only

RESULTS_FOLDER = "boundResults"
OUTPUT_FOLDER = "paperResults"

# Figures are drawn at their printed size: include them with width=\textwidth
FIG_WIDTH = 6.5         # inches
PANEL_HEIGHT = 2.3      # inches
FONT_SIZE = 9
LINE_WIDTH = 1.5
RGAP_TICKS = [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]   # RGAP axis on top of the iteration figures
RGAP_TICK_MIN_GAP = 0.2    # a tick closer than this (fraction of the panel width) to the previous one is left out

# LaTeX for all text if it is installed, otherwise matplotlib's LaTeX-like math text
if shutil.which("latex"):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                         "text.latex.preamble": r"\usepackage{amsmath,amssymb}"})
    PERCENT = r"\%"
    FRACTION = r"\dfrac"     # full-size fraction in the legend (amsmath)
else:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    PERCENT = "%"
    FRACTION = r"\frac"
plt.rcParams.update({"font.size": FONT_SIZE, "hatch.linewidth": 0.5})

WARNINGS = []    # printed once at the end


# =============================================================================
# READING THE RUNS
# =============================================================================

def runFolder(network, theta):
    return Path(RESULTS_FOLDER) / network / f"theta{theta}_d{DEMAND_MULTIPLIER}x_K{NETWORKS[network]}_{STEP_RULE}"


def displayName(network):
    return DISPLAY_NAMES.get(network, network)


def firstBelow(rgap, level):
    """Position of the first iteration with RGAP <= level, or None if RGAP never got there."""
    positions = np.flatnonzero(rgap <= level)
    if positions.size == 0:
        return None
    return int(positions[0])


def readRun(network, theta):
    """
    Columns of plots.csv (in long double) up to the first iteration with RGAP <= BOUND_ACCURACY.
    A run that stopped earlier (at the time limit) is kept whole.
    """
    fileName = runFolder(network, theta) / "plots.csv"
    if not fileName.exists():
        WARNINGS.append(f"missing run: {fileName}")
        return None
    table = np.genfromtxt(fileName, delimiter=",", names=True, dtype=np.longdouble, ndmin=1)
    last = firstBelow(table["rgap"], BOUND_ACCURACY)
    if last is not None:
        table = table[:last + 1]
    return {column: table[column] for column in table.dtype.names}


def fromRgap(data, level):
    """All columns from the first iteration with RGAP <= level on, or None (also if data is None)."""
    row = None if data is None else firstBelow(data["rgap"], level)
    if row is None:
        return None
    return {column: values[row:] for column, values in data.items()}


def valuesAt(runs, network, theta, level):
    """All columns at the first iteration with RGAP <= level, or None."""
    data = runs[(network, theta)]
    if data is None:
        return None
    position = firstBelow(data["rgap"], level)
    if position is None:
        WARNINGS.append(f"RGAP {level:.0e} not reached: {network}, theta={theta}")
        return None
    return {column: data[column][position] for column in data}


# =============================================================================
# TABLES
# =============================================================================

def formatPower(x):
    """0.00012 -> $1\\times10^{-4}$ (one significant digit, as in the paper tables)."""
    x = float(x)
    if not np.isfinite(x):
        return ""
    if x == 0:
        return "$0$"
    mantissa, exponent = f"{x:.0e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def formatLevel(level):
    """1e-6 -> $10^{-6}$"""
    return f"$10^{{{int(round(np.log10(level)))}}}$"


def tableRow(cells):
    return " & ".join(cells) + r" \\"


def networksRows(runs):
    """tab:4_res: Network & OD pairs & Links & Nodes & Demand & Paths & Path set (from summary.json).
    The last two columns of the paper table (mean CV and mean Jaccard index) are not computed here."""
    lines = []
    for network in NETWORKS:
        summary = None
        for theta in THETAS:
            fileName = runFolder(network, theta) / "summary.json"
            if fileName.exists():
                summary = json.loads(fileName.read_text())
                break
        if summary is None:
            lines.append(tableRow([displayName(network)] + [""] * 6))
            continue
        lines.append(tableRow([displayName(network), f"{float(summary['od_pairs']):.0f}", f"{float(summary['links']):.0f}",
                               f"{float(summary['nodes']):.0f}", f"{float(summary['demand']):.0f}",
                               f"{float(summary['paths']):.0f}", "Yen"]))
    return lines


# tab:4_cost rows: (label, time column of plots.csv), mean seconds per iteration.
# The first row is one iteration of the algorithm including the bounds. Each bound row is the extra time of that bound.
COST_ROWS = [
    (r"Iteration of the algorithm", "iter_time"),
    (r"Objective, $\gamma(\BFh)$", "time_gamma"),
    (r"Path flow, $m_h$", "time_m_h"),
    (r"Link flow, $m_a^{l}$", "time_link_flow"),
    (r"Link cost, $L_a^{l}m_a^{l}$", "time_link_cost"),
    (r"TSTT", "time_tstt"),
]
# "All bounds" is the sum of the general bounds in the paper. The bound_time column of plots.csv is not
# used: it also times phi, the RGAP and the bounds with tilde m_a^l of Lemma lem:3_linkbound, which are not part of this table.
ALL_BOUNDS = ["time_gamma", "time_m_h", "time_link_flow", "time_link_cost", "time_tstt"]


def costIterations(runs, network):
    """
    Columns of the COST_THETA run of a network over iterations 1 to the first with RGAP <= the
    smallest table level, or None. Iteration 0 is left out: it has no iteration time.
    """
    data = runs[(network, COST_THETA)]
    last = None if data is None else firstBelow(data["rgap"], min(TABLE_RGAP_LEVELS))
    if last is None:
        return None
    return {column: np.asarray(values[1:last + 1], dtype=float) for column, values in data.items()}


def costRows(runs):
    """tab:4_cost: mean seconds per iteration, and the share of time in the bounds = all bounds / iteration (the iteration time includes the bounds)."""

    lines = []

    columns = {network: costIterations(runs, network) for network in COST_NETWORKS}
    for network in COST_NETWORKS:    # iteration time including the bounds
        columns[network]["iter_time"] = columns[network]["iter_time"] + sum(columns[network][column] for column in ALL_BOUNDS)


    for label, column in COST_ROWS:
        cells = [label]
        for network in COST_NETWORKS:
            cells.append(formatPower(np.mean(columns[network][column])))
        lines.append(tableRow(cells))
    allCells = ["All bounds"]
    shareCells = [r"Time in bounds (\%)"]
    for network in COST_NETWORKS:
        data = columns[network]
        boundSeconds = sum(data[column] for column in ALL_BOUNDS)    # per iteration
        allCells.append(formatPower(np.mean(boundSeconds)))
        share = 100 * np.sum(boundSeconds) / np.sum(data["iter_time"])
        shareCells.append(f"${share:.0f}$")
    lines.append(tableRow(allCells))
    lines.append(tableRow(shareCells))
    return lines


def objpathRows(runs):
    """tab:4_objpath: phi(h) - phi(hat h) and gamma(h) in percent of |phi(hat h)|, ||h - hat h|| and m_h in
    percent of ||hat h||. Two-word names over two rows."""
    lines = []
    for network in NETWORKS:
        words = displayName(network).split(" ")
        nameRows = [displayName(network)] if len(words) == 1 else [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
        for position, level in enumerate(TABLE_RGAP_LEVELS):
            cells = [nameRows[position] if position < len(nameRows) else "", formatLevel(level)]
            for theta in THETAS:
                values = valuesAt(runs, network, theta, level)
                if values is None:
                    cells += [""] * 4
                else:
                    toObjective = 100 / abs(float(values["hat_phi"]))
                    toPath = 100 / float(values["hat_h_l2"])
                    cells += [formatPower(toObjective * values["delta_phi"]), formatPower(toObjective * values["gamma"]),
                              formatPower(toPath * values["error_h_l2"]), formatPower(toPath * values["m_h"])]
            lines.append(tableRow(cells))
        lines.append(r"\addlinespace")
    return lines[:-1]    # no \addlinespace after the last network


# tab:4_link quantities: (label, true error column, bound column, equilibrium norm column)
LINK_ROWS = [
    ("Link flow", "error_a_l2", "M_L", "hat_a_l2"),
    ("Link cost", "error_T_l2", "M_C", "hat_T_l2"),
]


def linkRows(runs):
    """tab:4_link: true error and bound in percent of the equilibrium value (norms over all links)."""
    lines = []
    for network in NETWORKS:
        for quantity, (label, errorColumn, boundColumn, equilibriumColumn) in enumerate(LINK_ROWS):
            for position, level in enumerate(TABLE_RGAP_LEVELS):
                cells = [displayName(network) if quantity == 0 and position == 0 else "",
                         label if position == 0 else "", formatLevel(level)]
                for theta in THETAS:
                    values = valuesAt(runs, network, theta, level)
                    if values is None:
                        cells += ["", ""]
                        continue
                    toPercent = 100 / float(values[equilibriumColumn])
                    cells += [formatPower(toPercent * values[errorColumn]), formatPower(toPercent * values[boundColumn])]
                lines.append(tableRow(cells))
        lines.append(r"\addlinespace")
    return lines[:-1]


def tsttRows(runs):
    """tab:4_tstt: true error and bound in percent of zeta(hat a)."""
    lines = []
    for network in NETWORKS:
        for position, level in enumerate(TABLE_RGAP_LEVELS):
            cells = [displayName(network) if position == 0 else "", formatLevel(level)]
            for theta in THETAS:
                values = valuesAt(runs, network, theta, level)
                if values is None:
                    cells += ["", ""]
                    continue
                toPercent = 100 / abs(float(values["hat_zeta"]))
                cells += [formatPower(toPercent * values["error_zeta"]), formatPower(toPercent * values["m_zeta"])]
            lines.append(tableRow(cells))
        lines.append(r"\addlinespace")
    return lines[:-1]


# tab:4_convergence columns: (bound column, power). Each cell is the smallest c with
# bound <= c ||h - hat h||^power at every iterate of the run (Theorem thm:boundconvergence).
CONVERGENCE_COLUMNS = [
    ("gamma", 2),     # objective bound gamma(h)
    ("m_h", 1),       # path flow bound
    ("M_L", 1),       # link flow bound, norm over all links
    ("m_zeta", 1),    # TSTT bound
]


def convergenceRows(runs):
    """tab:4_convergence: Network & c for gamma, m_h, link flow and TSTT, at CONVERGENCE_TABLE_THETA."""
    lines = []
    for network in NETWORKS:
        cells = [displayName(network)]
        data = fromRgap(runs[(network, CONVERGENCE_TABLE_THETA)], CONVERGENCE_START_RGAP)
        for column, power in CONVERGENCE_COLUMNS:
            constant = None if data is None else envelopeConstant(data["error_h_l2"], data[column], power)
            cells.append("" if constant is None else formatPower(constant))
        lines.append(tableRow(cells))
    return lines


def writeTables(runs):
    lines = [
        "% Rows for the tables of Section sec:results, written by genPaperPlots.py.",
        "% Paste each block between \\midrule and \\bottomrule of its table.",
        f"% theta columns: {', '.join(str(t) for t in THETAS)}; "
        f"RGAP rows: {', '.join(f'{level:.0e}' for level in TABLE_RGAP_LEVELS)}.",
        "",
        "% ===== tab:4_res: Networks and path sets (add the CV and J columns by hand) =====", *networksRows(runs), "",
        "% ===== tab:4_cost: Computational cost of the bounds =====", *costRows(runs), "",
        "% ===== tab:4_objpath: Objective and path flow bounds =====", *objpathRows(runs), "",
        "% ===== tab:4_link: Link flow and link cost bounds =====", *linkRows(runs), "",
        "% ===== tab:4_tstt: TSTT bounds =====", *tsttRows(runs), "",
        f"% ===== tab:4_convergence: c with bound <= c ||h - hat h||^p (p = 2 for gamma, 1 otherwise), theta = {CONVERGENCE_TABLE_THETA} =====",
        "% columns: Network & objective gamma(h) & path flow m_h & link flow & TSTT",
        *convergenceRows(runs), "",
    ]
    (Path(OUTPUT_FOLDER) / "table.tex").write_text("\n".join(lines) + "\n")


# =============================================================================
# FIGURES
# =============================================================================

def logCurve(panel, data, values, style, label, color):
    """One curve against iteration. Values that are not positive (round-off at convergence) are left out."""
    values = np.asarray(values, dtype=float)
    values = np.where(values > 0, values, np.nan)
    zorder = 3 if style == "-" else 4    # solid lines at the back, so they never hide the others
    line, = panel.plot(np.asarray(data["iteration"], dtype=float), values, style, color=color,
                       linewidth=LINE_WIDTH, label=label, zorder=zorder)
    return line


def addRgapAxis(panel, data, verticalLines=False):
    """
    RGAP on a top axis: a tick at the first iteration where RGAP <= each level in RGAP_TICKS.
    verticalLines=True also draws a light dotted vertical line at each of these iterations.
    """
    rgap = np.asarray(data["rgap"], dtype=float)
    ticks, labels = [], []
    for level in RGAP_TICKS:
        row = firstBelow(rgap, level)
        if row is None:
            continue
        if row == 0 and ticks and ticks[-1] == float(data["iteration"][0]):
            # several levels already met at the first shown iteration: label only the smallest
            ticks.pop()
            labels.pop()
        left, right = panel.get_xlim()
        if ticks and float(data["iteration"][row]) - ticks[-1] < RGAP_TICK_MIN_GAP * (right - left):
            continue    # too close to the previous label
        ticks.append(float(data["iteration"][row]))
        labels.append(formatLevel(level))
    if verticalLines:
        for tick in ticks:
            panel.axvline(tick, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
    top = panel.twiny()
    top.set_xlim(panel.get_xlim())
    top.set_xticks(ticks)
    top.set_xticklabels(labels)
    top.set_xlabel("RGAP")


def saveFigure(figure, axes, handles, yLabel, fileName, xLabel="Iteration"):
    """Axis labels, one common legend in one line below the panels, and save."""
    for panel in axes:
        panel.set_xlabel(xLabel)
        if xLabel == "Iteration":
            panel.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0].set_ylabel(yLabel)
    figure.tight_layout()
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    figure.savefig(Path(OUTPUT_FOLDER) / fileName, bbox_inches="tight")
    plt.close(figure)


def sandwichFigure(runs):
    """Figure 1: objective sandwich G <= phi(h) - phi(hat h) <= gamma(h), one panel per network."""
    figure, axes = plt.subplots(1, len(FIGURE_NETWORKS), figsize=(FIG_WIDTH, PANEL_HEIGHT))
    for panel, network in zip(axes, FIGURE_NETWORKS):
        data = runs[(network, FIGURE_THETA)]
        handles = [
            logCurve(panel, data, data["objective_growth"], "-.",
                     rf"${FRACTION}{{1}}{{2\theta}}\sum_{{w\in\mathbb{{W}}}}{FRACTION}{{\|\mathbf{{h}}^{{w}}-\hat{{\mathbf{{h}}}}^{{w}}\|_1^2}}{{d_w}}$", "C0"),
            logCurve(panel, data, data["delta_phi"], "--", r"$\phi(\mathbf{h})-\phi(\hat{\mathbf{h}})$", "C1"),
            logCurve(panel, data, data["gamma"], "-", r"$\gamma(\mathbf{h})$", "C2"),
        ]
        panel.set_yscale("log")
        panel.set_title(displayName(network))
        addRgapAxis(panel, data, verticalLines=True)
    saveFigure(figure, axes, handles, "", f"objective_sandwich_theta_{FIGURE_THETA}.pdf")


# Panels of figures 2 and 3: (title, true error column, bound column, bound column with tilde m_a^l of
# Lemma lem:3_linkbound, equilibrium value column)
BOUND_PANELS = [
    ("Link flow", "error_a_l2", "M_L", "M_L_lemma", "hat_a_l2"),
    ("Link cost", "error_T_l2", "M_C", "M_C_lemma", "hat_T_l2"),
    ("TSTT", "error_zeta", "m_zeta", "m_zeta_lemma", "hat_zeta"),
]


def boundsFigure(runs, network):
    """Figures 2 and 3: true error, bound, and bound with tilde m_a^l of Lemma lem:3_linkbound, in percent of the equilibrium value, for one network."""
    data = runs[(network, FIGURE_THETA)]
    figure, axes = plt.subplots(1, len(BOUND_PANELS), figsize=(FIG_WIDTH, PANEL_HEIGHT))
    for panel, (title, errorColumn, boundColumn, lemmaColumn, equilibriumColumn) in zip(axes, BOUND_PANELS):
        toPercent = 100 / abs(float(data[equilibriumColumn][0]))
        handles = [
            logCurve(panel, data, toPercent * data[boundColumn], "-", "Error bound", "C1"),
            logCurve(panel, data, toPercent * data[lemmaColumn], "-.", r"Error bound with $\tilde{m}_a^{l}$", "C2"),
            logCurve(panel, data, toPercent * data[errorColumn], "--", "True error", "C0"),
            panel.axhline(1, color="black", linestyle=":", linewidth=LINE_WIDTH, zorder=2,
                          label=f"1{PERCENT} relative error"),
        ]
        panel.set_yscale("log")
        low, high = panel.get_ylim()
        panel.set_ylim(min(low, 0.5), max(high, 2))    # keep the 1% line visible
        panel.set_title(title)
        addRgapAxis(panel, data, verticalLines=True)
    fileName = f"bounds_{network.replace(' ', '')}_theta_{FIGURE_THETA}.pdf"
    saveFigure(figure, axes, handles, f"Relative error ({PERCENT})", fileName)


def percentFrom(values, hatZeta):
    """(values - zeta(hat a)) / zeta(hat a) x 100"""
    return 100 * (np.asarray(values, dtype=float) - hatZeta) / abs(hatZeta)


def tsttValueFigure(runs):
    """Figure 4: TSTT, the two-sided bound and the convex lower bound, on a symmetric log axis."""
    figure, axes = plt.subplots(1, len(FIGURE_NETWORKS), figsize=(FIG_WIDTH, PANEL_HEIGHT))
    for panel, network in zip(axes, FIGURE_NETWORKS):
        data = runs[(network, FIGURE_THETA)]
        x = np.asarray(data["iteration"], dtype=float)
        hatZeta = float(data["hat_zeta"][0])
        tstt = percentFrom(data["zeta"], hatZeta)
        upper = percentFrom(data["zeta"] + data["m_zeta"], hatZeta)
        lower = percentFrom(data["zeta"] - data["m_zeta"], hatZeta)
        convex = percentFrom(data["zeta_lower_convex"], hatZeta)

        # Two-sided bound: blue band with its two ends as lines
        panel.fill_between(x, lower, upper, color="C0", alpha=0.15, linewidth=0)
        panel.plot(x, upper, color="C0", linewidth=LINE_WIDTH)
        panel.plot(x, lower, color="C0", linewidth=LINE_WIDTH)
        # Convex lower bound: hatching from zeta(a) down to it, and the bound as a dashed line
        panel.fill_between(x, convex, tstt, facecolor="none", hatch="///", linewidth=0.5,
                           edgecolor=to_rgba("C1", 0.5))
        panel.plot(x, convex, "--", color="C1", linewidth=LINE_WIDTH)
        # TSTT on top of everything, and the equilibrium TSTT at 0
        panel.plot(x, tstt, color="black", linewidth=LINE_WIDTH, zorder=5)
        panel.axhline(0, color="black", linestyle=":", linewidth=LINE_WIDTH)
        # +1% and -1% error (grey, so they differ from the equilibrium TSTT line)
        for level in (1, -1):
            panel.axhline(level, color="0.5", linestyle=":", linewidth=LINE_WIDTH, zorder=2)

        # Symmetric log axis: linear within one decade below the smallest bound, log outside;
        # ticks at 0 and at most 3 powers of ten on each side; +/-1% always visible
        finite = np.isfinite(upper) & np.isfinite(lower)     # a bound is inf or nan where a path flow underflowed to 0
        smallest = max(np.min(np.abs(upper - tstt)[finite]), 1e-12)
        linear = 10.0 ** (np.floor(np.log10(smallest)) - 1)
        panel.set_yscale("symlog", linthresh=linear, linscale=1)
        top = 1.5 * max(np.max(np.abs(upper[finite])), np.max(np.abs(lower[finite])), linear, 1)
        panel.set_ylim(-top, top)
        low = int(np.log10(linear)) + 1
        high = max(int(np.floor(np.log10(top))), low)
        powers = range(high, low - 1, -max(1, int(np.ceil((high - low + 1) / 3))))
        panel.set_yticks([-(10.0 ** p) for p in powers] + [0] + [10.0 ** p for p in powers])
        panel.yaxis.set_minor_locator(NullLocator())

        panel.set_title(displayName(network))
        addRgapAxis(panel, data)

    handles = [
        Patch(facecolor=to_rgba("C0", 0.15), edgecolor="C0", linewidth=LINE_WIDTH, label="Bound"),
        Patch(facecolor="none", hatch="///", edgecolor="C1", linestyle="--", linewidth=LINE_WIDTH, label="Convex bound"),
        Line2D([], [], color="black", linewidth=LINE_WIDTH, label="TSTT"),
        Line2D([], [], color="black", linestyle=":", linewidth=LINE_WIDTH, label="Equilibrium TSTT"),
        Line2D([], [], color="0.5", linestyle=":", linewidth=LINE_WIDTH, label=rf"$\pm$1{PERCENT} error"),
    ]
    saveFigure(figure, axes, handles, "Percentage error", f"tstt_value_theta_{FIGURE_THETA}.pdf")


# Convergence (Theorem thm:boundconvergence): gamma(h) falls like ||h - hat h||^2 and the
# other bounds like ||h - hat h||. Each panel shows the bound and ||h - hat h||^power against iteration.
ERROR_LABELS = {    # power -> legend label of ||h - hat h||^power
    1: r"$\|\mathbf{h}-\hat{\mathbf{h}}\|$",
    2: r"$\|\mathbf{h}-\hat{\mathbf{h}}\|^2$",
}
ERROR_STYLES = {1: "--", 2: "-."}
ERROR_COLORS = {1: "C0", 2: "C2"}
# Panels of the convergence figure: (title, bound column, power)
CONVERGENCE_PANELS = [
    ("Objective bound", "gamma", 2),
    ("Link flow bound", "M_L", 1),
    ("TSTT bound", "m_zeta", 1),
]


def positivePairs(x, y):
    """x and y where both are finite and positive (the points a log-log plot can draw)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[keep], y[keep]


def envelopeConstant(x, y, power):
    """Smallest c with y <= c x^power at every point, or None. Example: x=[1, 0.1], y=[2, 0.05], power 2 -> 5"""
    x, y = positivePairs(x, y)
    return float(np.max(y / x ** power)) if x.size else None


def convergenceFigure(runs, network):
    """Objective, link flow and TSTT bound, each with ||h - hat h||^2 or ||h - hat h||, against iteration,
    from the first iteration with RGAP <= CONVERGENCE_START_RGAP on."""
    data = fromRgap(runs[(network, FIGURE_THETA)], CONVERGENCE_START_RGAP)
    if data is None:
        return
    figure, axes = plt.subplots(1, len(CONVERGENCE_PANELS), figsize=(FIG_WIDTH, PANEL_HEIGHT))
    errorLines = {}
    for panel, (title, boundColumn, power) in zip(axes, CONVERGENCE_PANELS):
        bound = logCurve(panel, data, data[boundColumn], "-", "Bound", "C1")
        errorLines[power] = logCurve(panel, data, np.asarray(data["error_h_l2"], dtype=float) ** power,
                                     ERROR_STYLES[power], ERROR_LABELS[power], ERROR_COLORS[power])
        panel.set_yscale("log")
        panel.set_title(title)
        addRgapAxis(panel, data, verticalLines=True)
    handles = [bound] + [errorLines[power] for power in sorted(errorLines, reverse=True)]
    fileName = f"convergence_{network.replace(' ', '')}_theta_{FIGURE_THETA}.pdf"
    saveFigure(figure, axes, handles, "", fileName)


# =============================================================================
# MAIN
# =============================================================================

def makePaperPlots():
    """Write table.tex and the paper figures to OUTPUT_FOLDER from the runs in RESULTS_FOLDER."""
    WARNINGS.clear()
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    runs = {}
    for network in NETWORKS:
        for theta in set(THETAS + [COST_THETA, FIGURE_THETA]):
            runs[(network, theta)] = readRun(network, theta)

    writeTables(runs)

    if all(runs[(network, FIGURE_THETA)] is not None for network in FIGURE_NETWORKS):
        sandwichFigure(runs)
        tsttValueFigure(runs)
    for network in BOUND_FIGURE_NETWORKS:
        if runs[(network, FIGURE_THETA)] is not None:
            boundsFigure(runs, network)
    for network in CONVERGENCE_NETWORKS:
        if runs[(network, FIGURE_THETA)] is not None:
            convergenceFigure(runs, network)

    for warning in dict.fromkeys(WARNINGS):
        print("warning:", warning)
    print("Written to", OUTPUT_FOLDER)


if __name__ == "__main__":
    makePaperPlots()