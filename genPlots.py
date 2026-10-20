"""
genPlots.py

Makes every figure of every bound run written by SUEPath. It writes no tables and
nothing to paperResults/; genPaperPlots.py makes table.tex and the paper figures.

WHAT GOES IN
    For every (network, theta), SUEPath.StochasticUE.assignment(calcBounds=True)
    writes a run folder containing plots.csv, one row per iteration (written by recordBounds).

WHAT COMES OUT
    plots/<Network>/<run specs>/<plot>.pdf
        Every figure of every run, in the same folder layout as boundResults/, e.g.
        plots/Chicago Sketch/theta1.5_d1.0x_K20_BB1/link_flow_bound.pdf
    plots/<Network>/theta<theta>_algorithms/<plot>.pdf
        Path flow bound of every step rule in main.ALGORITHMS (MSA-HS, BB1, BB2)
        against ||h - hat h||, e.g. plots/Chicago Sketch/theta0.5_algorithms/path_flow_convergence.pdf

HOW TO RUN
    main.py calls makeRunPlots after solving.
    To redraw every figure from saved runs without solving again, run this file:
        python genPlots.py
    It reads every boundResults/<Network>/<run specs>/plots.csv. The path flow convergence
    figures of several step rules are drawn wherever runs of more than one step rule exist.

NOTATION (as in the paper)
    gamma(h)        objective bound, Theorem thm:3_objectivegap
    m_h             path flow bound, Theorem thm:3_globalpathbound
    tilde m_a^l     link flow bound before the strengthening, Lemma lem:3_linkbound
    m_a^l           link flow bound, Theorem thm:3_linkbound
    L_a^l m_a^l     link cost bound, Corollary cor:3_linkcostbound
    TSTT bound      Theorem thm:3_tsttbound
    Convex TSTT     one-sided lower bound, Corollary cor:3_tsttconvex
                    (shown only in the tstt_value_symlog figure)
    "with tilde m_a^l"  the link flow, link cost and TSTT bounds with tilde m_a^l in place of m_a^l
"""
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # draw to files only, no window
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter


# =============================================================================
# LATEX TEXT IN THE FIGURES
# =============================================================================
# If LaTeX is installed, every label is typeset by LaTeX (same font as the paper).
# If not, matplotlib's own math text with the Computer Modern font is used,
# which still looks LaTeX-like.

USE_LATEX = shutil.which("latex") is not None

if USE_LATEX:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,amssymb}"   # amssymb for \mathbb{W}
else:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    print("Warning: latex not found, falling back to mathtext (output still looks LaTeX-like).")

# The percent sign starts a comment in LaTeX, so LaTeX needs \% instead of %.
if USE_LATEX:
    PERCENT = r"\%"
else:
    PERCENT = "%"


# =============================================================================
# SETTINGS
# =============================================================================

# Figure size, line width and font sizes. The fonts are large because three
# figures share one row in the paper.
PLOT_CFG = {
    "figsize": (10, 8),
    "linewidth": 3,
    "xlabel_fontsize": 36,
    "ylabel_fontsize": 36,
    "xtick_fontsize": 30,
    "ytick_fontsize": 30,
    "legend_fontsize": 36,
}

# Horizontal axis of the figures: "iteration" or "rgap".
X_AXIS = "iteration"

# RGAP levels shown as ticks on the RGAP axis on top of the percent figures.
RGAP_TICKS = (1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10)

# Smallest distance between two RGAP tick labels, as a fraction of the axis width;
# a tick closer than this to the previous one is left out so the labels do not overlap.
RGAP_TICK_MIN_GAP = 0.14

# Space (points) between the "RGAP" label and the RGAP tick labels on top.
RGAP_LABEL_GAP = 15

# Space (inches) between the x axis label and the legend below it.
LEGEND_GAP = 0.15

# TSTT value figure (tstt_value_symlog.pdf)
BOUND_LINEWIDTH = 2           # lines of the General band ends and the convex lower bound
GENERAL_BAND_COLOR = "C0"     # shade of the General band
BAND_ALPHA = 0.15             # transparency of the General band shade
CONVEX_BAND_COLOR = "C1"      # hatching and dashed line of the convex lower bound
CONVEX_HATCH_ALPHA = 0.5      # transparency of the hatching
CONVEX_HATCH = "///"          # hatching of the convex region
plt.rcParams["hatch.linewidth"] = 0.6    # light hatching


# Convergence figures start at the first iteration with RGAP <= this: the convergence
# results (Theorem thm:boundconvergence) describe the behaviour close to equilibrium.
CONVERGENCE_START_RGAP = 1e-4

# Line styles used in the figures.
SOLID = "-"
DASHED = "--"
DASH_DOT = "-."
DOTTED = ":"


# =============================================================================
# 1. NAMES OF NETWORKS AND FILES
# =============================================================================

def displayName(networkName):
    """
    Name of a network as it appears in the folder names under plots/.
    Network names are the folder names in testNetworks/, e.g. "Chicago Sketch".
    An old-style name ending in " Network" is shortened the same way:
    "Chicago Sketch Network" -> "Chicago Sketch"
    """
    name = str(networkName).strip()
    if name.endswith(" Network"):
        name = name[:-len(" Network")]
    return name



def runPlotsFolder(plotsFolder, networkName, runFolder):
    """
    Folder of all figures of one run: plots/<Network>/<run specs>/.
    <run specs> is the name of the run folder in boundResults/, so both folders
    have the same layout.
    Example: ("plots", "Chicago Sketch", "boundResults/Chicago Sketch/theta1.5_d1.0x_K20_BB1")
             -> plots/Chicago Sketch/theta1.5_d1.0x_K20_BB1
    """
    runSpecs = Path(runFolder).name
    return Path(plotsFolder) / displayName(networkName) / runSpecs



# =============================================================================
# 2. READING A RUN
# =============================================================================

def readRun(runFolder, boundAccuracy):
    """
    Read plots.csv of one run.

    Rows are kept up to and including the first iteration with RGAP <= boundAccuracy,
    so every run ends at the same accuracy.

    Returns a dictionary: column name -> array with one value per iteration.
    Values are read in long double, the precision SUEPath writes them in.
    """
    table = np.genfromtxt(Path(runFolder) / "plots.csv", delimiter=",",
                          names=True, dtype=np.longdouble, ndmin=1)

    lastRow = firstIterationBelow(table["rgap"], boundAccuracy)
    if lastRow is not None:
        table = table[:lastRow + 1]

    data = {}
    for column in table.dtype.names:
        data[column] = table[column]
    return data



def firstIterationBelow(rgapValues, level):
    """Position of the first iteration with RGAP <= level, or None if RGAP never got there."""
    positions = np.flatnonzero(rgapValues <= level)
    if positions.size == 0:
        return None
    return int(positions[0])


def fromRgap(data, level):
    """All columns from the first iteration with RGAP <= level on, or None if RGAP never got there."""
    row = firstIterationBelow(data["rgap"], level)
    if row is None:
        return None
    return {column: values[row:] for column, values in data.items()}


def inPercent(values, referenceValue):
    """100 * values / |referenceValue|, used for errors relative to the equilibrium."""
    return 100 * np.asarray(values, dtype=float) / abs(float(referenceValue))


# =============================================================================
# 3. DRAWING ONE FIGURE
# =============================================================================

def applyPlotStyle(axes, xLabel, yLabel):
    """Axis labels and tick sizes, as in the old plotter."""
    axes.set_xlabel(xLabel, fontsize=PLOT_CFG["xlabel_fontsize"])
    axes.set_ylabel(yLabel, fontsize=PLOT_CFG["ylabel_fontsize"])
    axes.tick_params(axis="x", labelsize=PLOT_CFG["xtick_fontsize"])
    axes.tick_params(axis="y", labelsize=PLOT_CFG["ytick_fontsize"])
    axes.xaxis.get_offset_text().set_fontsize(PLOT_CFG["xtick_fontsize"])
    axes.yaxis.get_offset_text().set_fontsize(PLOT_CFG["ytick_fontsize"])
    axes.grid(False)


def addLegendBelow(figure, axes, curveLines):
    """
    Legend of the curves (curveLines, the lines returned by axes.plot) below the plot,
    on one line if it fits within the width of the figure, otherwise on as few lines as needed.

    It is centered under the whole figure (y label and tick labels included),
    not only under the plotting area, which would make it look shifted right.
    Call this after figure.tight_layout(), when everything else is in place.
    """
    # Box around everything drawn so far, in inches
    renderer = figure.canvas.get_renderer()
    contentBox = figure.get_tightbbox(renderer)

    # Top middle of the legend, as fractions of the figure width and height:
    # centered under the content, LEGEND_GAP inches below the x label.
    middle = (contentBox.x0 + contentBox.x1) / 2 / figure.get_figwidth()
    top = (contentBox.y0 - LEGEND_GAP) / figure.get_figheight()

    # One line if the legend fits within the width of the figure; otherwise as few lines as
    # needed, so a long legend does not make the figure wider (which shrinks its text in the paper)
    for columns in range(len(curveLines), 0, -1):
        legend = axes.legend(
            handles=curveLines,           # only the curves, not the 1% line
            fontsize=PLOT_CFG["legend_fontsize"],
            ncol=columns,
            loc="upper center",
            bbox_to_anchor=(middle, top),
            bbox_transform=figure.transFigure,
            handlelength=2.5,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        legendWidth = legend.get_window_extent(renderer).width / figure.dpi    # inches
        if legendWidth <= contentBox.width:
            break


def addOnePercentLine(axes):
    """
    Horizontal line at 1% error, with its own small legend in the top right corner
    of the plot (separate from the legend of the curves below the plot).
    The y axis is widened if needed so the line is always visible.
    """
    line = axes.axhline(1, color="black", linestyle=DOTTED, linewidth=2, zorder=2,
                        label=f"1{PERCENT} relative error")

    lower, upper = axes.get_ylim()
    if lower > 0.5:
        lower = 0.5
    if upper < 2:
        upper = 2
    axes.set_ylim(lower, upper)

    onePercentLegend = axes.legend(handles=[line], loc="upper right",
                                   fontsize=PLOT_CFG["legend_fontsize"], handlelength=2.5)
    # A later axes.legend(...) call replaces the legend of the axes, so keep this one
    # as a separate artist.
    axes.add_artist(onePercentLegend)


def addRgapAxis(axes, data):
    """
    Second x axis on top showing the RGAP.

    One tick for every level in RGAP_TICKS, placed at the first iteration where
    RGAP <= that level. Levels the run never reached get no tick.
    Example: RGAP first drops below 1e-4 at iteration 37 -> tick $10^{-4}$ above iteration 37.
    """
    iterations = np.asarray(data["iteration"], dtype=float)
    rgap = np.asarray(data["rgap"], dtype=float)
    left, right = axes.get_xlim()

    tickPositions = []
    tickLabels = []
    for level in RGAP_TICKS:
        row = firstIterationBelow(rgap, level)
        if row is None:             # this level was never reached
            continue
        if row == 0 and tickPositions and tickPositions[-1] == iterations[0]:
            # several levels already met at the first shown iteration: label only the smallest
            tickPositions.pop()
            tickLabels.pop()
        if tickPositions and iterations[row] - tickPositions[-1] < RGAP_TICK_MIN_GAP * (right - left):
            continue                # too close to the previous label
        tickPositions.append(iterations[row])
        tickLabels.append(formatLevel(level))     # e.g. 1e-4 -> "$10^{-4}$"

    topAxis = axes.twiny()
    topAxis.set_xlim(left, right)
    topAxis.set_xticks(tickPositions)
    topAxis.set_xticklabels(tickLabels)
    topAxis.set_xlabel("RGAP", fontsize=PLOT_CFG["xlabel_fontsize"], labelpad=RGAP_LABEL_GAP)
    topAxis.tick_params(axis="x", labelsize=PLOT_CFG["xtick_fontsize"])


def drawFigure(fileName, data, curves, yLabel, onePercentLine=False, rgapAxis=False):
    """
    Draw curves against iteration (or RGAP) on a logarithmic y axis and save as a pdf.

    curves          list of (values, legend label, line style)
                    Solid lines are drawn at the back, dashed and dash-dot lines on top,
                    so a solid line never hides the others.
    onePercentLine  True adds a line at 1% error (for the percent figures)
    rgapAxis        True adds the RGAP as a second x axis on top (only when X_AXIS is "iteration")

    Only positive values can be drawn on a log axis; zero or negative values
    (for example round-off right at convergence) are left out of the curve.
    """
    # Horizontal axis
    if X_AXIS == "rgap":
        x = np.asarray(data["rgap"], dtype=float)
        xLabel = "Relative gap"
    else:
        x = np.asarray(data["iteration"], dtype=float)
        xLabel = "Iteration"

    figure, axes = plt.subplots(figsize=PLOT_CFG["figsize"])

    # One line per curve. Solid lines at the back (zorder 3), all other lines on top (zorder 4).
    # Both are above the 1% line (zorder 2).
    curveLines = []
    anyPositiveValue = False
    for values, label, lineStyle in curves:
        if lineStyle == SOLID:
            drawingOrder = 3
        else:
            drawingOrder = 4
        values = np.broadcast_to(np.asarray(values, dtype=float), x.shape)
        canBeDrawn = np.isfinite(values) & (values > 0)
        if np.any(canBeDrawn):
            anyPositiveValue = True
        valuesToDraw = np.where(canBeDrawn, values, np.nan)

        marker = "o" if len(x) == 1 else None    # a single point needs a marker to be visible
        line, = axes.plot(x, valuesToDraw, label=label, linestyle=lineStyle, marker=marker,
                          linewidth=PLOT_CFG["linewidth"], zorder=drawingOrder)
        curveLines.append(line)

    if not anyPositiveValue:
        axes.set_ylim(1e-16, 1)
        axes.text(0.5, 0.5, "No positive values", transform=axes.transAxes,
                  ha="center", va="center", fontsize=PLOT_CFG["legend_fontsize"])

    # Vertical axis: logarithmic
    axes.set_yscale("log")
    axes.yaxis.set_major_locator(LogLocator(numticks=7))
    lower, upper = axes.get_ylim()
    if upper / lower < 10:
        # Less than one power of ten: plain numbers read better than powers of ten
        axes.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axes.yaxis.set_major_formatter(ScalarFormatter())
        axes.yaxis.set_minor_formatter(NullFormatter())    # no labels on the minor log ticks

    if onePercentLine:
        addOnePercentLine(axes)

    # Horizontal axis ticks
    if X_AXIS == "rgap":
        axes.set_xscale("log")
        axes.invert_xaxis()    # RGAP decreases from left to right
    else:
        axes.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=1))

    applyPlotStyle(axes, xLabel, yLabel)

    if rgapAxis and X_AXIS == "iteration":
        addRgapAxis(axes, data)

    figure.tight_layout()
    addLegendBelow(figure, axes, curveLines)
    figure.savefig(fileName, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# 4. ALL FIGURES OF ONE RUN
# =============================================================================

def plotRun(runFolder, plotsFolder, networkName, theta, boundAccuracy):
    """
    Write all figures of one (network, theta) run to plots/<Network>/<run specs>/.
    Every figure ends at the first iteration with RGAP <= boundAccuracy.

    Figures (file <plot>.pdf):
      objective_sandwich        G <= phi(h) - phi(hat h) <= gamma(h)
                                (Equation eq:3_globalpathsandwich, Figure fig:4_sandwich)
      path_flow_bound           path flow error and m_h, in percent of ||hat h||
      link_flow_bound           link flow error and bounds, in percent of ||hat a||
      link_cost_bound           link cost error and bounds, in percent of ||T(hat a)||
      tstt_bound                TSTT error and bounds, in percent of zeta(hat a)
                                (link flow, link cost and TSTT are Figure fig:4_linkbounds;
                                all four percent figures have a labelled line at 1% and
                                the RGAP on a second x axis)
      tstt_value_symlog         TSTT with the two-sided General bound and the one-sided convex
                                lower bound (Corollary cor:3_tsttconvex), as signed distance from
                                zeta(hat a) in percent of zeta(hat a), on a symmetric log axis;
      convergence_objective     gamma(h) and ||h - hat h||^2 against iteration (Theorem thm:boundconvergence)
      convergence_path_flow     m_h and ||h - hat h|| against iteration
      convergence_link_flow     ||m_a|| and ||h - hat h|| against iteration
      convergence_tstt          TSTT bound and ||h - hat h|| against iteration
      computation_time          seconds per iteration of the algorithm and of all bounds
    """
    outputFolder = runPlotsFolder(plotsFolder, networkName, runFolder)
    outputFolder.mkdir(parents=True, exist_ok=True)

    data = readRun(runFolder, boundAccuracy)

    def fileFor(plot):
        return outputFolder / f"{plot}.pdf"

    plotObjectiveSandwich(data, fileFor("objective_sandwich"))
    plotPathFlowBound(data, fileFor("path_flow_bound"))
    plotLinkFlowBound(data, fileFor("link_flow_bound"))
    plotLinkCostBound(data, fileFor("link_cost_bound"))
    plotTsttBound(data, fileFor("tstt_bound"))
    plotTsttValue(data, fileFor("tstt_value_symlog"))
    plotConvergence(data, "gamma", 2, "Objective bound", fileFor("convergence_objective"))
    plotConvergence(data, "m_h", 1, "Path flow bound", fileFor("convergence_path_flow"))
    plotConvergence(data, "M_L", 1, "Link flow bound", fileFor("convergence_link_flow"))
    plotConvergence(data, "m_zeta", 1, "TSTT bound", fileFor("convergence_tstt"))
    plotComputationTime(data, fileFor("computation_time"))


def plotObjectiveSandwich(data, fileName):
    """G <= phi(h) - phi(hat h) <= gamma(h)."""
    curves = [
        (data["objective_growth"],
         r"$\frac{1}{2\theta}\sum_{w\in\mathbb{W}}\frac{\|\mathbf{h}^{w}-\hat{\mathbf{h}}^{w}\|_1^2}{d_w}$",
         DASH_DOT),
        (data["delta_phi"], r"$\phi(\mathbf{h})-\phi(\hat{\mathbf{h}})$", DASHED),
        (data["gamma"], r"$\gamma(\mathbf{h})$", SOLID),
    ]
    drawFigure(fileName, data, curves, "Objective difference", rgapAxis=True)


def plotPathFlowBound(data, fileName):
    """
    In percent of ||hat h||:
      actual error  100 ||h - hat h|| / ||hat h||
      bound         100 m_h / ||hat h||        (Theorem thm:3_globalpathbound)
    """
    hatH = data["hat_h_l2"][0]
    curves = [
        (inPercent(data["error_h_l2"], hatH), "Actual error", DASHED),
        (inPercent(data["m_h"], hatH), "Bound", SOLID),
    ]
    drawFigure(fileName, data, curves, f"Path flow error ({PERCENT})",
               onePercentLine=True, rgapAxis=True)


def plotLinkFlowBound(data, fileName):
    """
    In percent of ||hat a||:
      actual error  100 ||a - hat a|| / ||hat a||
      Error bound   100 ||m_a|| / ||hat a||        with m_a = (m_a^l) over all links
      with tilde    100 ||tilde m_a|| / ||hat a||  with tilde m_a = (tilde m_a^l) over all links
    """
    hatA = data["hat_a_l2"][0]
    curves = [
        (inPercent(data["error_a_l2"], hatA), "Actual error", DASHED),
        (inPercent(data["M_L"], hatA), "Error bound", SOLID),
        (inPercent(data["M_L_lemma"], hatA), r"Error bound with $\tilde{m}_a^{l}$", DASH_DOT),
    ]
    drawFigure(fileName, data, curves, f"Link flow error ({PERCENT})",
               onePercentLine=True, rgapAxis=True)


def plotLinkCostBound(data, fileName):
    """
    In percent of ||T(hat a)||:
      actual error  100 ||T(a) - T(hat a)|| / ||T(hat a)||
      Error bound   100 ||(L_a^l m_a^l)|| / ||T(hat a)||
      with tilde    the same with tilde m_a^l in place of m_a^l
    """
    hatT = data["hat_T_l2"][0]
    curves = [
        (inPercent(data["error_T_l2"], hatT), "Actual error", DASHED),
        (inPercent(data["M_C"], hatT), "Error bound", SOLID),
        (inPercent(data["M_C_lemma"], hatT), r"Error bound with $\tilde{m}_a^{l}$", DASH_DOT),
    ]
    drawFigure(fileName, data, curves, f"Link cost error ({PERCENT})",
               onePercentLine=True, rgapAxis=True)


def plotTsttBound(data, fileName):
    """
    In percent of zeta(hat a):
      actual error  100 |zeta(a) - zeta(hat a)| / zeta(hat a)
      Error bound   100 B / zeta(hat a), B from Theorem thm:3_tsttbound
      with tilde    the same with tilde m_a^l in place of m_a^l
    """
    hatZeta = data["hat_zeta"][0]
    curves = [
        (inPercent(data["error_zeta"], hatZeta), "Actual error", DASHED),
        (inPercent(data["m_zeta"], hatZeta), "Error bound", SOLID),
        (inPercent(data["m_zeta_lemma"], hatZeta), r"Error bound with $\tilde{m}_a^{l}$", DASH_DOT),
    ]
    drawFigure(fileName, data, curves, f"TSTT error ({PERCENT})",
               onePercentLine=True, rgapAxis=True)


def plotTsttValue(data, fileName):
    """
    TSTT and its bounds as y = (value - zeta(hat a)) / zeta(hat a) x 100, on a symmetric log axis:
      zeta(a)        current TSTT                                    black line
      zeta(hat a)    equilibrium TSTT, at 0                          dotted line
      General bound  zeta(a) - B and zeta(a) + B (Theorem thm:3_tsttbound)       blue band
      Convex bound   zeta_lower_convex, lower end only (Corollary cor:3_tsttconvex)  orange hatching
    """
    hatZeta = float(data["hat_zeta"][0])

    def percent(values):
        """(values - zeta(hat a)) / zeta(hat a) x 100"""
        return 100 * (np.asarray(values, dtype=float) - hatZeta) / abs(hatZeta)

    tstt = percent(data["zeta"])
    upper = percent(data["zeta"] + data["m_zeta"])
    lower = percent(data["zeta"] - data["m_zeta"])
    convex = percent(data["zeta_lower_convex"])

    x = np.asarray(data["iteration"], dtype=float)

    figure, axes = plt.subplots(figsize=PLOT_CFG["figsize"])

    # General bound: blue band with its two ends as lines
    axes.fill_between(x, lower, upper, color=GENERAL_BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    axes.plot(x, upper, color=GENERAL_BAND_COLOR, linewidth=BOUND_LINEWIDTH)
    axes.plot(x, lower, color=GENERAL_BAND_COLOR, linewidth=BOUND_LINEWIDTH)

    # Convex bound: hatching from its lower end up to zeta(a), and the lower end as a dashed line
    # (matplotlib leaves out hatching without an outline, hence linewidth=0.5)
    axes.fill_between(x, convex, tstt, facecolor="none", hatch=CONVEX_HATCH, linewidth=0.5,
                      edgecolor=to_rgba(CONVEX_BAND_COLOR, CONVEX_HATCH_ALPHA))
    axes.plot(x, convex, color=CONVEX_BAND_COLOR, linestyle=DASHED, linewidth=BOUND_LINEWIDTH)

    # zeta(a) on top of everything, and zeta(hat a)
    axes.plot(x, tstt, color="black", linewidth=PLOT_CFG["linewidth"], zorder=5, label=r"$\zeta(\mathbf{a})$")
    axes.axhline(0, color="black", linestyle=DOTTED, linewidth=2, label=r"$\zeta(\hat{\mathbf{a}})$")

    # Symmetric log y axis: linear within one decade below the smallest bound, log outside.
    # Ticks at 0 and +/- powers of ten (at most 4 per side), kept clear of the label of 0.
    # Limits from finite values only (a bound is inf or nan where a path flow underflowed to 0)
    finite = np.isfinite(upper) & np.isfinite(lower)
    smallest = max(np.min(np.abs(upper - tstt)[finite]), 1e-12)     # B in percent, kept above 0 for the log
    linear = 10.0 ** (np.floor(np.log10(smallest)) - 1)
    axes.set_yscale("symlog", linthresh=linear, linscale=1)
    top = 1.5 * max(np.max(np.abs(upper[finite])), np.max(np.abs(lower[finite])), linear)
    axes.set_ylim(-top, top)
    low, high = int(np.log10(linear)) + 1, max(int(np.floor(np.log10(top))), int(np.log10(linear)) + 1)
    powers = range(high, low - 1, -max(1, int(np.ceil((high - low + 1) / 4))))
    axes.set_yticks([-(10.0 ** p) for p in powers] + [0] + [10.0 ** p for p in powers])
    axes.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=1))

    fraction = r"\dfrac" if USE_LATEX else r"\frac"     # \dfrac (amsmath) draws it full size
    applyPlotStyle(axes, "Iteration",
                   rf"${fraction}{{\zeta(\mathbf{{a}})-\zeta(\hat{{\mathbf{{a}}}})}}{{\zeta(\hat{{\mathbf{{a}}}})}}\times 100$")

    addRgapAxis(axes, data)

    # Legend of zeta(a) and zeta(hat a) inside the plot, bottom right (kept as a separate
    # artist, because the legend below replaces the legend of the axes)
    axes.add_artist(axes.legend(loc="lower right", fontsize=PLOT_CFG["legend_fontsize"], handlelength=2.5))

    # Legend of the two bounds below the plot, as in the other figures
    figure.tight_layout()
    generalBox = Patch(facecolor=to_rgba(GENERAL_BAND_COLOR, BAND_ALPHA), edgecolor=GENERAL_BAND_COLOR,
                       linewidth=BOUND_LINEWIDTH, label="General bound (two-sided)")
    convexBox = Patch(facecolor="none", hatch=CONVEX_HATCH, edgecolor=CONVEX_BAND_COLOR,
                      linestyle=DASHED, linewidth=BOUND_LINEWIDTH, label="Convex bound (lower only)")
    addLegendBelow(figure, axes, [generalBox, convexBox])
    figure.savefig(fileName, bbox_inches="tight")
    plt.close(figure)


# =============================================================================
# 4b. CONVERGENCE OF THE BOUNDS
# =============================================================================
# Theorem thm:boundconvergence: gamma(h) = O(||L(h) - h||^2) and the other bounds are
# O(||L(h) - h||), and ||L(h) - h|| <= (1 + L) ||h - hat h||. So gamma falls like
# ||h - hat h||^2 and the other bounds like ||h - hat h||, whatever algorithm produced h.

H_ERROR_LABEL = r"$\|\mathbf{h}-\hat{\mathbf{h}}\|$"


def positivePairs(x, y):
    """x and y (as float) where both are finite and positive, the only points a log-log plot can draw."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[keep], y[keep]


def drawLogLog(fileName, curves, yLabel, xLabel=H_ERROR_LABEL, minimumDecades=0):
    """
    Curves against ||h - hat h|| on log-log axes, saved as a pdf.
    The x axis runs from large to small error, so convergence goes from left to right.

    curves      list of (x values, y values, legend label, line style)
    minimumDecades  the y axis shows at least this many powers of ten (for a nearly constant ratio)
    """
    figure, axes = plt.subplots(figsize=PLOT_CFG["figsize"])

    curveLines = []
    anyPoint = False
    for x, y, label, lineStyle in curves:
        x, y = positivePairs(x, y)
        anyPoint = anyPoint or x.size > 0
        marker = "o" if len(x) == 1 else None
        line, = axes.plot(x, y, label=label, linestyle=lineStyle, marker=marker,
                          linewidth=PLOT_CFG["linewidth"], zorder=3 if lineStyle == SOLID else 4)
        curveLines.append(line)

    if not anyPoint:
        axes.text(0.5, 0.5, "No positive values", transform=axes.transAxes,
                  ha="center", va="center", fontsize=PLOT_CFG["legend_fontsize"])

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.invert_xaxis()
    low, high = axes.get_ylim()
    if minimumDecades > 0 and high / low < 10 ** minimumDecades:
        middle = np.sqrt(low * high)
        axes.set_ylim(middle / 10 ** (minimumDecades / 2), middle * 10 ** (minimumDecades / 2))
    applyPlotStyle(axes, xLabel, yLabel)

    figure.tight_layout()
    addLegendBelow(figure, axes, curveLines)
    figure.savefig(fileName, bbox_inches="tight")
    plt.close(figure)


def plotConvergence(data, boundColumn, power, boundLabel, fileName):
    """
    One bound and ||h - hat h||^power against iteration (from RGAP <= CONVERGENCE_START_RGAP on),
    on a log axis, so their slopes can be compared:
      gamma   with ||h - hat h||^2  (objective bound)
      m_h     with ||h - hat h||    (path flow bound)
      M_L     with ||h - hat h||    (link flow bound, norm over all links)
      m_zeta  with ||h - hat h||    (TSTT bound)
    """
    data = fromRgap(data, CONVERGENCE_START_RGAP)
    if data is None:
        return
    if power == 2:
        errorLabel = r"$\|\mathbf{h}-\hat{\mathbf{h}}\|^2$"
    else:
        errorLabel = r"$\|\mathbf{h}-\hat{\mathbf{h}}\|$"
    curves = [
        (data[boundColumn], boundLabel, SOLID),
        (data["error_h_l2"] ** power, errorLabel, DASHED),
    ]
    drawFigure(fileName, data, curves, "", rgapAxis=True)


def plotComputationTime(data, fileName):
    """
    Seconds per iteration: one iteration of the algorithm (iter_time, without the bounds)
    and all bounds at that iterate (bound_time). Row 0 has no iteration time, so it is left out.
    """
    later = np.asarray(data["iteration"], dtype=float) > 0
    shown = {column: values[later] for column, values in data.items()}
    if len(shown["iteration"]) == 0:
        return
    curves = [
        (shown["iter_time"], "Iteration of the algorithm", DASHED),
        (shown["bound_time"], "All bounds", SOLID),
    ]
    drawFigure(fileName, shown, curves, "Time (s)", rgapAxis=True)


# =============================================================================
# 4c. PATH FLOW BOUND OF SEVERAL ALGORITHMS
# =============================================================================

ALGORITHM_STYLES = (SOLID, DASHED, DASH_DOT, DOTTED)


def plotAlgorithmComparison(runFolders, plotsFolder, networkName, theta, boundAccuracy):
    """
    m_h against ||h - hat h|| for several step rules on one (network, theta), written to
    plots/<Network>/theta<theta>_algorithms/:
      path_flow_convergence   m_h of every algorithm against ||h - hat h||
      path_flow_ratio         m_h / ||h - hat h||, which stays bounded if m_h = O(||h - hat h||)

    runFolders  {algorithm label: run folder}, e.g. {"MSA-HS": ..., "BB1": ..., "BB2": ...}
    Every run is drawn from RGAP <= CONVERGENCE_START_RGAP on.
    """
    outputFolder = Path(plotsFolder) / displayName(networkName) / f"theta{theta}_algorithms"
    outputFolder.mkdir(parents=True, exist_ok=True)

    curves = []
    ratioCurves = []
    for position, (label, runFolder) in enumerate(runFolders.items()):
        data = fromRgap(readRun(runFolder, boundAccuracy), CONVERGENCE_START_RGAP)
        if data is None:
            continue
        style = ALGORITHM_STYLES[position % len(ALGORITHM_STYLES)]
        x = np.asarray(data["error_h_l2"], dtype=float)
        y = np.asarray(data["m_h"], dtype=float)
        curves.append((x, y, label, style))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratioCurves.append((x, y / x, label, style))

    drawLogLog(outputFolder / "path_flow_convergence.pdf", curves, "Path flow bound")
    drawLogLog(outputFolder / "path_flow_ratio.pdf", ratioCurves,
               r"Path flow bound / $\|\mathbf{h}-\hat{\mathbf{h}}\|$", minimumDecades=2)


# =============================================================================
# 5. NUMBER FORMATTING FOR THE RGAP AXIS
# =============================================================================

def formatLevel(level):
    """An RGAP level as $10^{b}$. Example: 1e-6 -> "$10^{-6}$" """
    exponent = int(round(np.log10(level)))
    return f"$10^{{{exponent}}}$"


# =============================================================================
# 6. REDRAW EVERY FIGURE FROM SAVED RUNS (python genPlots.py)
# =============================================================================

def makeRunPlots(resultsFolder="boundResults", plotsFolder="plots", boundAccuracy=1e-8):
    """
    Draw every figure of every run in resultsFolder into plotsFolder.
    Run folders are named theta<theta>_d<demand multiplier>x_K<k>_<step rule>.
    boundAccuracy is the RGAP of the bound runs in main.py; the figures end there.
    """
    import re
    runName = re.compile(r"theta(?P<theta>[^_]+)_d(?P<demand>[^_]+)x_K(?P<k>\d+)_(?P<rule>.+)")

    # Runs that differ only in the step rule, for the path flow convergence figures
    comparisons = {}

    for csvFile in sorted(Path(resultsFolder).glob("*/*/plots.csv")):
        runFolder = csvFile.parent
        networkName = runFolder.parent.name
        match = runName.fullmatch(runFolder.name)
        if match is None:
            continue
        print("Drawing", runFolder)
        plotRun(runFolder, plotsFolder, networkName, float(match["theta"]), boundAccuracy)
        key = (networkName, match["theta"], match["demand"], match["k"])
        comparisons.setdefault(key, {})[match["rule"]] = runFolder

    for (networkName, theta, demand, k), runs in comparisons.items():
        if len(runs) > 1:
            plotAlgorithmComparison(runs, plotsFolder, networkName, theta, boundAccuracy)

    print("Figures written to", Path(plotsFolder).resolve())


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent)
    makeRunPlots()