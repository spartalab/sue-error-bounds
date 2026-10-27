"""
genPaperPlots.py

Writes the four figures and table.tex of Section sec:results from the bound runs that SUEPath
writes to boundResults/<Network>/theta<theta>_d<d>x_K<k>_<stepRule>/plots.csv.
All settings are hardcoded below.  main.py calls makePaperPlots after solving; to redo only
the figures and tables from saved runs, run:  python genPaperPlots.py

Output, in OUTPUT_FOLDER:
    objective_sandwich_theta_<theta>.pdf         Sioux Falls | Winnipeg | Chicago Sketch
    bounds_ChicagoSketch_theta_<theta>.pdf       Link flow | Link cost | TSTT
    tstt_value_theta_<theta>.pdf                 Sioux Falls | Winnipeg | Chicago Sketch
    convergence_ChicagoSketch_theta_<theta>.pdf  Objective | Link flow | TSTT bound, with ||h - hat h||(^2)
    table.tex                                    one block per table; paste each between \\midrule and \\bottomrule
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
from matplotlib.ticker import FixedFormatter, MaxNLocator, NullLocator


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
THETAS = [0.5, 1.5]                     # one group of table columns each
STEP_RULE = "BB1"
DEMAND_MULTIPLIER = 1.0
BOUND_ACCURACY = 1e-8                   # every run is cut at the first iteration with RGAP <= this
TABLE_RGAP_LEVELS = [1e-4, 1e-6, 1e-8]  # rows of the tables

FIGURE_THETA = 0.5                      # theta of the figures, tab:4_cost and tab:4_convergence
FIGURE_NETWORKS = ["Sioux Falls", "Winnipeg", "Chicago Sketch"]     # panels of the objective sandwich and TSTT value figures
BOUND_FIGURE_NETWORK = "Chicago Sketch"                             # network of the bounds and convergence figures
COST_NETWORKS = ["Anaheim", "Winnipeg", "Chicago Sketch"]           # columns of tab:4_cost
CONVERGENCE_START_RGAP = 1e-4           # tab:4_convergence uses RGAP <= this only

RESULTS_FOLDER = "boundResults"
OUTPUT_FOLDER = "paperResults"

# Figures are drawn at their printed size: include them with width=\textwidth
FIG_WIDTH = 6.5         # inches
PANEL_HEIGHT = 2.3      # inches
FONT_SIZE = 9
LINE_WIDTH = 1.5
RGAP_TICKS = [1e0, 1e-2, 1e-4, 1e-6, 1e-8]   # RGAP axis on top of every panel
RGAP_LABEL_WIDTH = 0.19    # about the width of an RGAP label plus a small gap, as a fraction of the panel width

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


# =============================================================================
# HELPERS
# =============================================================================

def firstBelow(rgap, level):
    """Position of the first iteration with RGAP <= level, or None if RGAP never got there."""
    positions = np.flatnonzero(rgap <= level)
    return int(positions[0]) if positions.size else None


def formatPower(x):
    """0.00012 -> $1\\times10^{-4}$ (one significant digit, as in the paper tables). inf or nan -> empty cell."""
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


def logCurve(panel, data, values, style, label, color):
    """One curve against iteration. Values that are not positive (round-off at convergence) are left out."""
    values = np.asarray(values, dtype=float)
    values = np.where(values > 0, values, np.nan)
    zorder = 3 if style == "-" else 4    # solid lines at the back, so they never hide the others
    line, = panel.plot(np.asarray(data["iteration"], dtype=float), values, style, color=color,
                       linewidth=LINE_WIDTH, label=label, zorder=zorder)
    return line


def addRgapAxis(panel, data, verticalLines=True):
    """
    RGAP on a top axis: a tick at the first iteration with RGAP <= each level in RGAP_TICKS, and with
    verticalLines a light dotted vertical line at each of these iterations. A label is centered on its tick
    unless it would run into the previous label. Then it starts at its tick, and the previous label, if
    centered and with room on its left, ends at its tick.
    """
    rgap = np.asarray(data["rgap"], dtype=float)
    ticks, labels = [], []
    for level in RGAP_TICKS:
        row = firstBelow(rgap, level)
        if row is not None:
            ticks.append(float(data["iteration"][row]))
            labels.append(formatLevel(level))
    if verticalLines:
        for tick in np.unique(ticks):
            panel.axvline(tick, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
    top = panel.twiny()
    top.set_xlim(panel.get_xlim())
    top.set_xticks(ticks)
    top.xaxis.set_major_formatter(FixedFormatter(labels))    # labels by position, so two levels can share a tick
    texts = top.get_xticklabels()
    top.set_xlabel("RGAP")
    left, right = panel.get_xlim()
    width = RGAP_LABEL_WIDTH * (right - left)    # one label, in iterations
    ends = []                                    # right end of each label, in iterations
    for i, tick in enumerate(ticks):
        if ends and tick - width / 2 < ends[-1]:    # centered, it would run into the previous label
            if texts[i - 1].get_horizontalalignment() == "center" and (i == 1 or ticks[i - 1] - width >= ends[-2]):
                texts[i - 1].set_horizontalalignment("right")    # the previous label ends at its tick
                ends[-1] = ticks[i - 1]
        if ends and tick - width / 2 < ends[-1]:
            texts[i].set_horizontalalignment("left")             # this label starts at its tick
            ends.append(tick + width)
        else:
            ends.append(tick + width / 2)


def saveFigure(figure, axes, handles, yLabel, fileName):
    """Axis labels, one common legend in one line below the panels, and save."""
    for panel in axes:
        panel.set_xlabel("Iteration")
        panel.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0].set_ylabel(yLabel)
    figure.tight_layout()
    figure.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    figure.savefig(Path(OUTPUT_FOLDER) / fileName, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# FIGURES AND TABLES
# =============================================================================

def makePaperPlots():
    """Write the four figures and table.tex to OUTPUT_FOLDER from the runs in RESULTS_FOLDER."""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Read the runs
    # runs[(network, theta)]: columns of plots.csv in long double, up to the first iteration with
    # RGAP <= BOUND_ACCURACY (a run that stopped earlier, at the time limit, is kept whole); None if missing.
    # summaries[network]: summary.json of the first theta that has one.
    # -------------------------------------------------------------------------
    runs, summaries = {}, {}
    for network in NETWORKS:
        for theta in THETAS:
            folder = Path(RESULTS_FOLDER) / network / f"theta{theta}_d{DEMAND_MULTIPLIER}x_K{NETWORKS[network]}_{STEP_RULE}"
            if network not in summaries and (folder / "summary.json").exists():
                summaries[network] = json.loads((folder / "summary.json").read_text())
            if (folder / "plots.csv").exists():
                table = np.genfromtxt(folder / "plots.csv", delimiter=",", names=True, dtype=np.longdouble, ndmin=1)
                last = firstBelow(table["rgap"], BOUND_ACCURACY)
                if last is not None:
                    table = table[:last + 1]
                runs[(network, theta)] = {column: table[column] for column in table.dtype.names}
            else:
                print("warning: missing run:", folder / "plots.csv")
                runs[(network, theta)] = None

    # -------------------------------------------------------------------------
    # Objective sandwich G <= phi(h) - phi(hat h) <= gamma(h), one panel per network
    # -------------------------------------------------------------------------
    if all(runs[(network, FIGURE_THETA)] is not None for network in FIGURE_NETWORKS):
        growthLabel = rf"${FRACTION}{{1}}{{2\theta}}\sum_{{w\in\mathbb{{W}}}}{FRACTION}{{\|\mathbf{{h}}^{{w}}-\hat{{\mathbf{{h}}}}^{{w}}\|_1^2}}{{d_w}}$"
        figure, axes = plt.subplots(1, len(FIGURE_NETWORKS), figsize=(FIG_WIDTH, PANEL_HEIGHT))
        for panel, network in zip(axes, FIGURE_NETWORKS):
            data = runs[(network, FIGURE_THETA)]
            handles = [
                logCurve(panel, data, data["objective_growth"], "-.", growthLabel, "C0"),
                logCurve(panel, data, data["delta_phi"], "--", r"$\phi(\mathbf{h})-\phi(\hat{\mathbf{h}})$", "C1"),
                logCurve(panel, data, data["gamma"], "-", r"$\gamma(\mathbf{h})$", "C2"),
            ]
            panel.set_yscale("log")
            panel.set_title(network)
            addRgapAxis(panel, data)
        saveFigure(figure, axes, handles, "", f"objective_sandwich_theta_{FIGURE_THETA}.pdf")

    # -------------------------------------------------------------------------
    # Link flow, link cost and TSTT bounds of BOUND_FIGURE_NETWORK. True error, bound, and bound with
    # tilde m_a^l of Lemma lem:3_linkbound, in percent of the equilibrium value.
    # -------------------------------------------------------------------------
    data = runs[(BOUND_FIGURE_NETWORK, FIGURE_THETA)]
    if data is not None:
        panels = [    # (title, true error column, bound column, bound column with tilde m_a^l, equilibrium value column)
            ("Link flow", "error_a_l2", "M_L", "M_L_lemma", "hat_a_l2"),
            ("Link cost", "error_T_l2", "M_C", "M_C_lemma", "hat_T_l2"),
            ("TSTT", "error_zeta", "m_zeta", "m_zeta_lemma", "hat_zeta"),
        ]
        figure, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH, PANEL_HEIGHT))
        for panel, (title, errorColumn, boundColumn, lemmaColumn, equilibriumColumn) in zip(axes, panels):
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
            addRgapAxis(panel, data)
        saveFigure(figure, axes, handles, f"Relative error ({PERCENT})",
                   f"bounds_{BOUND_FIGURE_NETWORK.replace(' ', '')}_theta_{FIGURE_THETA}.pdf")

    # -------------------------------------------------------------------------
    # TSTT value: TSTT, the two-sided bound and the convex lower bound, in percent from the equilibrium TSTT
    # zeta(hat a), on a symmetric log axis, one panel per network
    # -------------------------------------------------------------------------
    if all(runs[(network, FIGURE_THETA)] is not None for network in FIGURE_NETWORKS):
        figure, axes = plt.subplots(1, len(FIGURE_NETWORKS), figsize=(FIG_WIDTH, PANEL_HEIGHT))
        for panel, network in zip(axes, FIGURE_NETWORKS):
            data = runs[(network, FIGURE_THETA)]
            x = np.asarray(data["iteration"], dtype=float)
            hatZeta = float(data["hat_zeta"][0])
            tstt = 100 * (np.asarray(data["zeta"], dtype=float) - hatZeta) / abs(hatZeta)
            upper = 100 * (np.asarray(data["zeta"] + data["m_zeta"], dtype=float) - hatZeta) / abs(hatZeta)
            lower = 100 * (np.asarray(data["zeta"] - data["m_zeta"], dtype=float) - hatZeta) / abs(hatZeta)
            convex = 100 * (np.asarray(data["zeta_lower_convex"], dtype=float) - hatZeta) / abs(hatZeta)

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

            panel.set_title(network)
            addRgapAxis(panel, data, verticalLines=False)
        handles = [
            Patch(facecolor=to_rgba("C0", 0.15), edgecolor="C0", linewidth=LINE_WIDTH, label="Bound"),
            Patch(facecolor="none", hatch="///", edgecolor="C1", linestyle="--", linewidth=LINE_WIDTH,
                  label="Convex bound"),
            Line2D([], [], color="black", linewidth=LINE_WIDTH, label="TSTT"),
            Line2D([], [], color="black", linestyle=":", linewidth=LINE_WIDTH, label="Equilibrium TSTT"),
            Line2D([], [], color="0.5", linestyle=":", linewidth=LINE_WIDTH, label=rf"$\pm$1{PERCENT} error"),
        ]
        saveFigure(figure, axes, handles, "Percentage error", f"tstt_value_theta_{FIGURE_THETA}.pdf")

    # -------------------------------------------------------------------------
    # Convergence of the bounds of BOUND_FIGURE_NETWORK (Theorem thm:boundconvergence). gamma(h) falls like
    # ||h - hat h||^2 and the other bounds like ||h - hat h||. The whole run, so the RGAP axis starts at 10^0.
    # -------------------------------------------------------------------------
    data = runs[(BOUND_FIGURE_NETWORK, FIGURE_THETA)]
    if data is not None:
        error = np.asarray(data["error_h_l2"], dtype=float)    # ||h - hat h||
        panels = [    # (title, bound column, power p, then style, color and label of ||h - hat h||^p)
            ("Objective bound", "gamma", 2, "-.", "C2", r"$\|\mathbf{h}-\hat{\mathbf{h}}\|^2$"),
            ("Link flow bound", "M_L", 1, "--", "C0", r"$\|\mathbf{h}-\hat{\mathbf{h}}\|$"),
            ("TSTT bound", "m_zeta", 1, "--", "C0", r"$\|\mathbf{h}-\hat{\mathbf{h}}\|$"),
        ]
        figure, axes = plt.subplots(1, len(panels), figsize=(FIG_WIDTH, PANEL_HEIGHT))
        errorLines = {}    # power -> line of ||h - hat h||^power
        for panel, (title, boundColumn, power, style, color, label) in zip(axes, panels):
            bound = logCurve(panel, data, data[boundColumn], "-", "Bound", "C1")
            errorLines[power] = logCurve(panel, data, error ** power, style, label, color)
            panel.set_yscale("log")
            panel.set_title(title)
            addRgapAxis(panel, data)
        saveFigure(figure, axes, [bound, errorLines[2], errorLines[1]], "",
                   f"convergence_{BOUND_FIGURE_NETWORK.replace(' ', '')}_theta_{FIGURE_THETA}.pdf")

    # -------------------------------------------------------------------------
    # Tables, written to table.tex
    # at[(network, theta, level)]: all columns at the first iteration with RGAP <= level, or None
    # -------------------------------------------------------------------------
    at = {}
    for (network, theta), data in runs.items():
        for level in TABLE_RGAP_LEVELS:
            row = None if data is None else firstBelow(data["rgap"], level)
            if data is not None and row is None:
                print(f"warning: RGAP {level:.0e} not reached: {network}, theta={theta}")
            at[(network, theta, level)] = None if row is None else {column: values[row] for column, values in data.items()}

    lines = [
        "% Rows for the tables of Section sec:results, written by genPaperPlots.py.",
        "% Paste each block between \\midrule and \\bottomrule of its table.",
        f"% theta columns: {', '.join(str(t) for t in THETAS)}; "
        f"RGAP rows: {', '.join(f'{level:.0e}' for level in TABLE_RGAP_LEVELS)}.",
        "",
    ]

    # tab:4_res: Network & OD pairs & Links & Nodes & Demand & Paths & Path set, from summary.json.
    # The last two columns of the paper table (mean CV and mean Jaccard index) are not computed here.
    lines.append("% ===== tab:4_res: Networks and path sets (add the CV and J columns by hand) =====")
    for network in NETWORKS:
        if network in summaries:
            cells = [f"{float(summaries[network][key]):.0f}" for key in ["od_pairs", "links", "nodes", "demand", "paths"]]
            cells.append("Yen")
        else:
            cells = [""] * 6
        lines.append(tableRow([network] + cells))
    lines.append("")

    # tab:4_cost: mean seconds per iteration at FIGURE_THETA, over iterations 1 (iteration 0 has no iteration time)
    # to the first with RGAP <= the smallest table level. The iteration includes the bounds, and each bound row is
    # the extra time of that bound. "All bounds" is the sum of the general bounds in the paper (the bound_time column
    # of plots.csv is not used: it also times phi, the RGAP and the bounds with tilde m_a^l of Lemma lem:3_linkbound).
    lines.append("% ===== tab:4_cost: Computational cost of the bounds =====")
    costRows = [    # (row label, time column of plots.csv)
        (r"Iteration of the algorithm", "iter_time"),
        (r"Objective, $\gamma(\BFh)$", "time_gamma"),
        (r"Path flow, $m_h$", "time_m_h"),
        (r"Link flow, $m_a^{l}$", "time_link_flow"),
        (r"Link cost, $L_a^{l}m_a^{l}$", "time_link_cost"),
        (r"TSTT", "time_tstt"),
    ]
    seconds = {}    # network -> {time column: seconds of each iteration}
    for network in COST_NETWORKS:
        data = runs[(network, FIGURE_THETA)]
        last = firstBelow(data["rgap"], min(TABLE_RGAP_LEVELS))
        times = {column: np.asarray(data[column][1:last + 1], dtype=float) for _, column in costRows}
        times["all_bounds"] = sum(times[column] for _, column in costRows[1:])
        times["iter_time"] = times["iter_time"] + times["all_bounds"]    # iteration including the bounds
        seconds[network] = times
    for label, column in costRows + [("All bounds", "all_bounds")]:
        lines.append(tableRow([label] + [formatPower(np.mean(seconds[network][column])) for network in COST_NETWORKS]))
    shares = [100 * np.sum(seconds[network]["all_bounds"]) / np.sum(seconds[network]["iter_time"]) for network in COST_NETWORKS]
    lines.append(tableRow([r"Time in bounds (\%)"] + [f"${share:.0f}$" for share in shares]))
    lines.append("")

    # tab:4_objpath: phi(h) - phi(hat h) and gamma(h) in percent of |phi(hat h)|, ||h - hat h|| and m_h in percent
    # of ||hat h||. A name of two or more words goes over two rows.
    lines.append("% ===== tab:4_objpath: Objective and path flow bounds =====")
    for network in NETWORKS:
        words = network.split(" ")
        half = len(words) // 2
        nameRows = [" ".join(words[:half]), " ".join(words[half:])] if half else [network]
        for position, level in enumerate(TABLE_RGAP_LEVELS):
            cells = [nameRows[position] if position < len(nameRows) else "", formatLevel(level)]
            for theta in THETAS:
                values = at[(network, theta, level)]
                if values is None:
                    cells += [""] * 4
                    continue
                toObjective = 100 / abs(float(values["hat_phi"]))
                toPath = 100 / float(values["hat_h_l2"])
                cells += [formatPower(toObjective * values["delta_phi"]), formatPower(toObjective * values["gamma"]),
                          formatPower(toPath * values["error_h_l2"]), formatPower(toPath * values["m_h"])]
            lines.append(tableRow(cells))
        lines.append(r"\addlinespace")
    lines[-1] = ""    # no \addlinespace after the last network

    # tab:4_link: true error and bound in percent of the equilibrium value (norms over all links)
    lines.append("% ===== tab:4_link: Link flow and link cost bounds =====")
    linkQuantities = [    # (label, true error column, bound column, equilibrium norm column)
        ("Link flow", "error_a_l2", "M_L", "hat_a_l2"),
        ("Link cost", "error_T_l2", "M_C", "hat_T_l2"),
    ]
    for network in NETWORKS:
        for quantity, (label, errorColumn, boundColumn, equilibriumColumn) in enumerate(linkQuantities):
            for position, level in enumerate(TABLE_RGAP_LEVELS):
                cells = [network if quantity == 0 and position == 0 else "", label if position == 0 else "",
                         formatLevel(level)]
                for theta in THETAS:
                    values = at[(network, theta, level)]
                    if values is None:
                        cells += ["", ""]
                        continue
                    toPercent = 100 / float(values[equilibriumColumn])
                    cells += [formatPower(toPercent * values[errorColumn]), formatPower(toPercent * values[boundColumn])]
                lines.append(tableRow(cells))
        lines.append(r"\addlinespace")
    lines[-1] = ""    # no \addlinespace after the last network

    # tab:4_tstt: true error and bound in percent of zeta(hat a)
    lines.append("% ===== tab:4_tstt: TSTT bounds =====")
    for network in NETWORKS:
        for position, level in enumerate(TABLE_RGAP_LEVELS):
            cells = [network if position == 0 else "", formatLevel(level)]
            for theta in THETAS:
                values = at[(network, theta, level)]
                if values is None:
                    cells += ["", ""]
                    continue
                toPercent = 100 / abs(float(values["hat_zeta"]))
                cells += [formatPower(toPercent * values["error_zeta"]), formatPower(toPercent * values["m_zeta"])]
            lines.append(tableRow(cells))
        lines.append(r"\addlinespace")
    lines[-1] = ""    # no \addlinespace after the last network

    # tab:4_convergence (Theorem thm:boundconvergence): the smallest c with bound <= c ||h - hat h||^p at every
    # iteration with RGAP <= CONVERGENCE_START_RGAP, p = 2 for gamma(h) and 1 for the others. Iterations where
    # the bound or ||h - hat h|| is not finite and positive are left out.
    lines.append(f"% ===== tab:4_convergence: c with bound <= c ||h - hat h||^p (p = 2 for gamma, 1 otherwise), "
                 f"theta = {FIGURE_THETA} =====")
    lines.append("% columns: Network & objective gamma(h) & path flow m_h & link flow & TSTT")
    for network in NETWORKS:
        cells = [network]
        data = runs[(network, FIGURE_THETA)]
        start = None if data is None else firstBelow(data["rgap"], CONVERGENCE_START_RGAP)
        for column, power in [("gamma", 2), ("m_h", 1), ("M_L", 1), ("m_zeta", 1)]:
            if start is None:
                cells.append("")
                continue
            x = np.asarray(data["error_h_l2"][start:], dtype=float)
            y = np.asarray(data[column][start:], dtype=float)
            keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            cells.append(formatPower(np.max(y[keep] / x[keep] ** power)) if keep.any() else "")
        lines.append(tableRow(cells))
    lines.append("")

    (Path(OUTPUT_FOLDER) / "table.tex").write_text("\n".join(lines) + "\n")
    print("Written to", OUTPUT_FOLDER)


if __name__ == "__main__":
    makePaperPlots()