"""Symbolic audit of Appendix sec:A_tightness (Tightness of the bounds) of the paper.
"""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import re
import sys


def remove_tex_comments(text):
    """Remove unescaped percent comments, preserving original line numbers."""
    active = []
    for line in text.splitlines():
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            previous = index - 1
            while previous >= 0 and line[previous] == "\\":
                backslashes += 1
                previous -= 1
            if backslashes % 2 == 0:
                line = line[:index]
                break
        active.append(line.rstrip())
    return "\n".join(active).strip()


class Audit:
    def __init__(self, sympy, variable, report_path, source, precision_order):
        self.s = sympy
        self.r = variable
        self.report_path = report_path
        self.source = source
        self.precision_order = precision_order
        self.rows = []
        self.output = []
        self.references = set()

    def emit(self, line=""):
        self.output.append(line)
        print(line, flush=True)

    def record(self, status, location, claim, evidence):
        self.references.update(re.findall(r"eq:[A-Za-z0-9_:]+", location))
        self.rows.append((status, location, claim, evidence))
        self.emit(f"STEP {len(self.rows):03d} {status}: {claim}")
        self.emit(f"  Source: {location}")
        self.emit(f"  Evidence: {evidence}")
        return status

    @lru_cache(maxsize=None)
    def simplify(self, expression):
        # force=True is deliberately never used.
        return self.s.simplify(self.s.expand_log(expression))

    @lru_cache(maxsize=None)
    def limit(self, expression):
        return self.s.limit(expression, self.r, 0, dir="+")

    def equal(self, location, claim, actual, expected):
        try:
            residual = self.simplify(actual - expected)
            if residual == 0:
                return self.record("PASS", location, claim, "simplify(actual - expected) = 0")
            if residual.is_zero is False:
                return self.record("FAIL", location, claim, f"nonzero residual = {residual}")
            # A provably nonzero value at the interior point r=1/4 is an
            # exact symbolic counterexample, not a numerical approximation.
            # It is used only to disprove an identity, never to prove one.
            witness = self.simplify(residual.subs(self.r, self.s.Rational(1, 4)))
            if not witness.free_symbols and witness.is_zero is False:
                return self.record("FAIL", location, claim,
                                   f"exact residual at r=1/4 is {witness}")
            return self.record("UNVERIFIED", location, claim, f"unresolved residual = {residual}")
        except Exception as exc:
            return self.record("UNVERIFIED", location, claim, f"{type(exc).__name__}: {exc}")

    def big_o(self, location, claim, exact, polynomial, order):
        """A finite limit of remainder/r**order rigorously proves the O claim."""
        try:
            quotient = (exact - polynomial) / self.r**order
            value = self.limit(quotient)
            evidence = f"limit((exact - ({polynomial}))/r**{order}, r, 0+) = {value}"
            if value.is_finite is True:
                return self.record("PASS", location, claim, evidence)
            if value.is_infinite is True or value in (self.s.oo, -self.s.oo, self.s.zoo):
                return self.record("FAIL", location, claim, evidence)
            return self.record("UNVERIFIED", location, claim, evidence)
        except Exception as exc:
            return self.record("UNVERIFIED", location, claim, f"{type(exc).__name__}: {exc}")

    def tends(self, location, claim, expression, expected):
        try:
            value = self.limit(expression)
            residual = self.simplify(value - expected)
            if residual == 0:
                status = "PASS"
            elif residual.is_zero is False and not value.has(self.s.Limit):
                status = "FAIL"
            else:
                status = "UNVERIFIED"
            return self.record(status, location, claim,
                               f"SymPy limit = {value}; expected = {expected}")
        except Exception as exc:
            return self.record("UNVERIFIED", location, claim, f"{type(exc).__name__}: {exc}")

    def positive(self, location, claim, expression, substitutions):
        try:
            value = self.simplify(expression.subs(substitutions, simultaneous=True))
            if value.is_positive is True:
                status = "PASS"
            elif value.is_positive is False:
                status = "FAIL"
            else:
                status = "UNVERIFIED"
            return self.record(status, location, claim,
                               f"after domain parameterization: {value}; is_positive={value.is_positive}")
        except Exception as exc:
            return self.record("UNVERIFIED", location, claim, f"{type(exc).__name__}: {exc}")

    def missing(self, location, claim, reason):
        return self.record("UNVERIFIED", location, claim, reason)

    def finish(self, series_expressions):
        self.emit()
        self.emit("EXACT SERIES COMPUTED BY SYMPY, FOR ADDITIONAL INSPECTION")
        for name, expression in series_expressions.items():
            try:
                self.emit(f"{name} = {self.s.series(expression, self.r, 0, self.precision_order)}")
            except Exception as exc:
                self.emit(f"{name}: series unavailable ({type(exc).__name__}: {exc})")
        self.emit()
        expected_labels = set(re.findall(r"\\label\{(eq:[^}]+)\}", self.source))
        missing_labels = sorted(expected_labels - self.references)
        self.emit("COVERAGE")
        self.emit(f"Active displayed equation labels: {len(expected_labels)}")
        self.emit(f"Labels linked to checks: {len(expected_labels & self.references)}")
        self.emit(f"Uncovered labels: {missing_labels}")
        if missing_labels:
            self.missing("source coverage", "All displayed equations have associated checks",
                         ", ".join(missing_labels))
        counts = Counter(row[0] for row in self.rows)
        self.emit()
        self.emit(f"SUMMARY: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
                  f"{counts['UNVERIFIED']} UNVERIFIED; {len(self.rows)} total steps")
        self.emit("Definitions and parameter choices are inputs, not independently proved facts.")
        self.emit("Source-label coverage does not turn missing external theorems into verified statements.")
        self.emit("Link 1's zero cost error and zero bound are checked separately; no 0/0 ratio is asserted.")
        self.report_path.write_text("\n".join(self.output) + "\n", encoding="utf-8")
        return 1 if counts["FAIL"] else (2 if counts["UNVERIFIED"] else 0)


if __name__ == "__main__":
    # All file paths, parameters, and precision settings are hardcoded here.
    report_path = Path(__file__).resolve().with_name("tightness_sympy_results.txt")
    source_path = Path(__file__).resolve().with_name("tightness_appendix.tex")    # optional: the appendix pasted from the paper
    precision_order = 8
    source_snapshot = r'''\section{Tightness of the bounds}\label{sec:A_tightness}

In this appendix, we ask whether the bounds could be reduced by a fixed factor for all problems satisfying our assumptions. To show that they cannot, it suffices to construct a family of problems in which the ratio of each error to its bound can be made arbitrarily close to one. To this end, we will use the network in Figure~\ref{fig:twolink}, with a single OD pair $w$ of demand $d_w=1$ served by two parallel links. We number the two paths so that path $i$ consists only of link $i$, and hence $a_1=h_1$ and $a_2=h_2$. Further, $d_{\max}=d_w=1$. Since each link is used by exactly one of the two paths, $\mathbb{W}_l=\{w\}$ and $S_l=1$ for both links. We also set $\theta=1$. We will look at a family of networks indexed by $r\in(0,1/2)$. We will first compute the flow errors and $\gamma(\BFh)$, which appears in every bound. We will then take each bound in turn, compute it together with the corresponding error, and show that their ratio tends to one as $r\to0$.

\begin{figure}[H]
\centering
\begin{tikzpicture}[>=stealth,
  od/.style={circle,draw,thick,minimum size=9mm,inner sep=0pt}]
\node[od] (o) at (0,0) {};
\node[od] (d) at (5,0) {};
\draw[->,thick] (o) to[bend left=40]
  node[midway,above]{Link 1: $\tau_1(x)=1$} (d);
\draw[->,thick] (o) to[bend right=40]
  node[midway,below]{Link 2: $\tau_2(x)=1-r+2rx$} (d);
\end{tikzpicture}
\caption{Two-link network used to examine tightness. A single OD pair $w$ from node $o$ to node $d$, with demand $d_w=1$, is served by two parallel links, each of which forms one path.}
\label{fig:twolink}
\end{figure}

Suppose the link costs are $\tau_1(x)=1$ and $\tau_2(x)=1-r+2rx$, where $x$ denotes the flow on the link. The path flows are the same as the link flows in this example. We observe that the equilibrium is $\hat h_1=\hat h_2=1/2$ for every $r$, because at these flows both links cost $\tau_1(1/2)=\tau_2(1/2)=1$, so the logit formula splits the demand equally between them. We take the current solution to be $h_1=1/2-r$ and $h_2=1/2+r$. The true errors at the current solution are
\begin{equation}
\begin{aligned}
&\|\BFh-\hat{\BFh}\|=\sqrt2\,r,\qquad\|\BFh^w-\hat{\BFh}^w\|_1=2r,\\
&|a_l-\hat a_l|=r,\qquad l\in\mathbb{L}.
\end{aligned}
\label{eq:appn:errors}
\end{equation}

We first compute $\gamma(\BFh)$. At the current flows, the path costs are $\mathcal{C}_1(\BFh)=1$ and $\mathcal{C}_2(\BFh)=1+2r^2$, so by Equation~\eqref{eq:2_suepathflow} the target flows are
\begin{equation}
\begin{aligned}
\mathcal{L}_1(\BFh)&=\frac{1}{1+e^{-2r^2}},\\
\mathcal{L}_2(\BFh)&=\frac{e^{-2r^2}}{1+e^{-2r^2}}=1-\mathcal{L}_1(\BFh).
\end{aligned}
\label{eq:appn:target}
\end{equation}
To expand these, we use $e^{-2r^2}=1-2r^2+O(r^4)$, so that $1+e^{-2r^2}=2(1-r^2)+O(r^4)$. Using $1/(1-r^2)=1+r^2+O(r^4)$ in Equation~\eqref{eq:appn:target} then gives
\begin{equation}
\begin{aligned}
\mathcal{L}_1(\BFh)&=\frac12\bigl(1+r^2\bigr)+O(r^4),\\
\mathcal{L}_2(\BFh)&=1-\mathcal{L}_1(\BFh)=\frac12-\frac{r^2}{2}+O(r^4).
\end{aligned}
\label{eq:appn:targetexp}
\end{equation}
From the definition of $\psi_h$ in Equation~\eqref{eq:3_fixedcostobjective}, $\psi_h(\BFh)=\beta(\BFh)+\eta(\BFh)$ and $\psi_h(\mathcal{L}(\BFh))=\beta(\BFh)+\mathcal{C}(\BFh)^T\bigl(\mathcal{L}(\BFh)-\BFh\bigr)+\eta(\mathcal{L}(\BFh))$. The term $\beta(\BFh)$ cancels, so
\begin{equation}
\begin{aligned}
\gamma(\BFh)&=\psi_h(\BFh)-\psi_h(\mathcal{L}(\BFh))\\
&=\mathcal{C}(\BFh)^T\bigl(\BFh-\mathcal{L}(\BFh)\bigr)+\eta(\BFh)-\eta(\mathcal{L}(\BFh)).
\end{aligned}
\label{eq:appn:gammadef}
\end{equation}
We compute the three terms separately. For the cost term, the current flows and the target flows in Equation~\eqref{eq:appn:targetexp} give
\begin{equation}
\begin{aligned}
h_1-\mathcal{L}_1(\BFh)&=\Bigl(\frac12-r\Bigr)-\Bigl(\frac12+\frac{r^2}{2}\Bigr)+O(r^4)\\
&=-r-\frac{r^2}{2}+O(r^4),\\
h_2-\mathcal{L}_2(\BFh)&=\Bigl(\frac12+r\Bigr)-\Bigl(\frac12-\frac{r^2}{2}\Bigr)+O(r^4)\\
&=r+\frac{r^2}{2}+O(r^4).
\end{aligned}
\label{eq:appn:diff}
\end{equation}
Substituting Equation~\eqref{eq:appn:diff} together with $\mathcal{C}_1(\BFh)=1$ and $\mathcal{C}_2(\BFh)=1+2r^2$, we obtain
\begin{equation}
\mathcal{C}(\BFh)^T\bigl(\BFh-\mathcal{L}(\BFh)\bigr)=2r^3+O(r^4).
\label{eq:appn:ch}
\end{equation}
Since $\theta=1$, the entropy at the current flows is
\begin{equation}
\begin{aligned}
\eta(\BFh)&=h_1\ln h_1+h_2\ln h_2\\
% &=\Bigl(\frac12-r\Bigr)\ln\Bigl(\frac12-r\Bigr)\\
% &\quad+\Bigl(\frac12+r\Bigr)\ln\Bigl(\frac12+r\Bigr)\\
% &=\Bigl(\frac12-r\Bigr)\bigl(-\ln2+\ln(1-2r)\bigr)\\
% &\quad+\Bigl(\frac12+r\Bigr)\bigl(-\ln2+\ln(1+2r)\bigr)\\
% &=\Bigl(\frac12-r\Bigr)\bigl(-\ln2-2r-2r^2\bigr)\\
% &\quad+\Bigl(\frac12+r\Bigr)\bigl(-\ln2+2r-2r^2\bigr)+O(r^3)\\
% &=\Bigl(-\frac12\ln2+r\ln2-r+r^2\Bigr)\\
% &\quad+\Bigl(-\frac12\ln2-r\ln2+r+r^2\Bigr)+O(r^3)\\
&=-\ln2+2r^2+O(r^3).
\end{aligned}
\label{eq:appn:nh}
\end{equation}
Similarly, substituting the target flows from Equation~\eqref{eq:appn:targetexp}, the entropy at the target flows is
\begin{equation}
\begin{aligned}
\eta(\mathcal{L}(\BFh))&=\mathcal{L}_1(\BFh)\ln\mathcal{L}_1(\BFh)\\
&\quad+\mathcal{L}_2(\BFh)\ln\mathcal{L}_2(\BFh)\\
% &=\Bigl(\frac12+\frac{r^2}{2}\Bigr)\ln\Bigl(\frac12+\frac{r^2}{2}\Bigr)\\
% &\quad+\Bigl(\frac12-\frac{r^2}{2}\Bigr)\ln\Bigl(\frac12-\frac{r^2}{2}\Bigr)+O(r^3)\\
% &=\Bigl(\frac12+\frac{r^2}{2}\Bigr)\bigl(-\ln2+\ln(1+r^2)\bigr)\\
% &\quad+\Bigl(\frac12-\frac{r^2}{2}\Bigr)\bigl(-\ln2+\ln(1-r^2)\bigr)+O(r^3)\\
% &=\Bigl(\frac12+\frac{r^2}{2}\Bigr)\bigl(-\ln2+r^2\bigr)\\
% &\quad+\Bigl(\frac12-\frac{r^2}{2}\Bigr)\bigl(-\ln2-r^2\bigr)+O(r^3)\\
% &=\Bigl(-\frac12\ln2-\frac{r^2}{2}\ln2+\frac{r^2}{2}\Bigr)\\
% &\quad+\Bigl(-\frac12\ln2+\frac{r^2}{2}\ln2-\frac{r^2}{2}\Bigr)+O(r^3)\\
&=-\ln2+O(r^3).
\end{aligned}
\label{eq:appn:nlh}
\end{equation}
Substituting Equations~\eqref{eq:appn:ch}, \eqref{eq:appn:nh}, and~\eqref{eq:appn:nlh} into Equation~\eqref{eq:appn:gammadef} gives
\begin{equation}
\begin{aligned}
\gamma(\BFh)&=2r^3+\bigl(-\ln2+2r^2\bigr)-\bigl(-\ln2\bigr)+O(r^3)\\
&=2r^2+O(r^3),
\end{aligned}
\label{eq:tight_gamma}
\end{equation}
so that $\gamma(\BFh)/r^2\to2$ as $r\to0$. We now examine each bound in turn.


\textit{Objective bound.} Since $\phi=\beta+\eta$, the actual error in the SUE objective is
\begin{equation}
\begin{aligned}
\phi(\BFh)-\phi(\hat{\BFh})&=\bigl(\beta(\BFh)-\beta(\hat{\BFh})\bigr)\\
&\quad+\bigl(\eta(\BFh)-\eta(\hat{\BFh})\bigr).
\end{aligned}
\label{eq:appn:objdecomp}
\end{equation}
For the cost term, the definition of $\beta$ in Equation~\eqref{eq:2_sueobjective} gives
\begin{equation}
\begin{aligned}
\beta(\BFh)&=\int_0^{h_1}1\,dx+\int_0^{h_2}(1-r+2rx)\,dx\\
&=h_1+(1-r)h_2+rh_2^2.
\end{aligned}
\label{eq:appn:beta}
\end{equation}
Substituting the current flows $h_1=\frac12-r$, $h_2=\frac12+r$ and the equilibrium flows $\hat h_1=\hat h_2=\frac12$ into Equation~\eqref{eq:appn:beta},
\begin{equation}
\begin{aligned}
\beta(\BFh)&=1-\frac r4+r^3,\\
\beta(\hat{\BFh})&=\frac12+\frac{1-r}{2}+\frac r4=1-\frac r4,
\end{aligned}
\label{eq:appn:betavals}
\end{equation}
so $\beta(\BFh)-\beta(\hat{\BFh})=r^3$. For the entropy term, $\eta(\hat{\BFh})=\frac12\ln\frac12+\frac12\ln\frac12=-\ln2$, and Equation~\eqref{eq:appn:nh} gives $\eta(\BFh)=-\ln2+2r^2+O(r^3)$, so $\eta(\BFh)-\eta(\hat{\BFh})=2r^2+O(r^3)$. Substituting both terms into Equation~\eqref{eq:appn:objdecomp} gives
\begin{equation}
\phi(\BFh)-\phi(\hat{\BFh})=r^3+2r^2+O(r^3)=2r^2+O(r^3).
\label{eq:appn:objerr}
\end{equation}
Since the bound in Theorem~\ref{thm:3_objectivegap} is $\gamma(\BFh)$, Equations~\eqref{eq:tight_gamma} and~\eqref{eq:appn:objerr} give the ratio
\begin{equation}
\frac{\phi(\BFh)-\phi(\hat{\BFh})}{\gamma(\BFh)}=\frac{2r^2+O(r^3)}{2r^2+O(r^3)}\to1.
\label{eq:appn:objratio}
\end{equation}


\textit{Path flow bounds.} The $\ell_2$ bound in Theorem~\ref{thm:3_globalpathbound} is $m_h=\sqrt{\theta d_{\max}\gamma(\BFh)}=\sqrt{\gamma(\BFh)}$, and the per-OD $\ell_1$ bound is $\sqrt{2\theta d_w\gamma(\BFh)}=\sqrt{2\gamma(\BFh)}$. Since $\gamma(\BFh)/r^2\to2$ by Equation~\eqref{eq:tight_gamma}, the path flow errors in Equation~\eqref{eq:appn:errors} give the two ratios
\begin{equation}
\begin{aligned}
\frac{\|\BFh-\hat{\BFh}\|}{m_h}&=\frac{\sqrt2\,r}{\sqrt{\gamma(\BFh)}}\to1,\\
\frac{\|\BFh^w-\hat{\BFh}^w\|_1}{\sqrt{2\gamma(\BFh)}}&=\frac{\sqrt2\,r}{\sqrt{\gamma(\BFh)}}\to1.
\end{aligned}
\label{eq:appn:pathratio}
\end{equation}

\textit{Link flow bound.} Since $S_l=1$, the bound in Lemma~\ref{lem:3_linkbound} is $\tilde m_a^{l}=\sqrt{\theta\gamma(\BFh)/2}=\sqrt{\gamma(\BFh)/2}$ for both links. Link 1 has a constant cost, so $R_a^{1}=0$ and the bound in Theorem~\ref{thm:3_linkbound} is $m_a^{1}=\tilde m_a^{1}=\sqrt{\gamma(\BFh)/2}$. For link 2, $\tau_2'(x)=2r$, so $R_a^{2}=2r$ and $m_a^{2}=\tilde m_a^{2}/\sqrt{1+r/2}=\sqrt{\gamma(\BFh)/(2+r)}$. With the link flow errors in Equation~\eqref{eq:appn:errors}, the ratios satisfy
\begin{equation}
\frac{|a_1-\hat a_1|}{m_a^{1}}=\frac{\sqrt2\,r}{\sqrt{\gamma(\BFh)}}\to1,
\qquad
\frac{|a_2-\hat a_2|}{m_a^{2}}=\frac{\sqrt{2+r}\,r}{\sqrt{\gamma(\BFh)}}\to1.
\label{eq:appn:linkratio}
\end{equation}

\textit{Link cost bound.} The cost of link 1 is constant, so both its error and its bound are zero. For link 2, Equation~\eqref{eq:appn:errors} gives the actual error $|\tau_2(a_2)-\tau_2(\hat a_2)|=2r|a_2-\hat a_2|=2r^2$. Since $\tau_2'(x)=2r$, Corollary~\ref{cor:3_linkcostbound} gives $L_a^{2}=2r$. Using $m_a^{2}=\sqrt{\gamma(\BFh)/(2+r)}$ from the link flow bound, the ratio satisfies
\begin{equation}
\begin{aligned}
\frac{|\tau_2(a_2)-\tau_2(\hat a_2)|}{L_a^{2}m_a^{2}}&=\frac{2r^2}{2r\sqrt{\gamma(\BFh)/(2+r)}}\\
&=\frac{\sqrt{2+r}\,r}{\sqrt{\gamma(\BFh)}}\to1.
\end{aligned}
\label{eq:appn:costratio}
\end{equation}

\textit{TSTT bound.} The TSTT is $\zeta(\BFa)=a_1+(1-r)a_2+2ra_2^2$, which equals $1$ at equilibrium and $1+r^2+2r^3$ at the current flows. The actual error is therefore $\zeta(\BFa)-\zeta(\hat{\BFa})=r^2+2r^3$. For the bound in Theorem~\ref{thm:3_tsttbound}, the marginal social path costs are $q_1=1$ and $q_2=\tau_2(a_2)+a_2\tau_2'(a_2)=1+r+4r^2$, so $Q_a=\sqrt{d_w}\,(q_2-q_1)=r+4r^2$. Moreover, since the cost of link 1 is constant and the cost of link 2 is linear, the second derivative of $\zeta$ with respect to each link flow does not depend on the flow, and equals $0$ for link 1 and $4r$ for link 2. Its maximum over $\mathbb{A}_a^l$ therefore gives $K_a^{1}=0$ and $K_a^{2}=4r$. Using $(m_a^{2})^2=\gamma(\BFh)/(2+r)$, the bound is therefore
\begin{equation}
\begin{aligned}
&Q_a\sqrt{\frac{\theta\gamma(\BFh)}{2}}+\frac12K_a^{2}(m_a^{2})^2\\
&\quad=(r+4r^2)\sqrt{\frac{\gamma(\BFh)}{2}}+\frac{2r\gamma(\BFh)}{2+r}.
\end{aligned}
\label{eq:appn:tsttbound}
\end{equation}
Dividing both the error and the bound in Equation~\eqref{eq:appn:tsttbound} by $r^2$, and using $\gamma(\BFh)/r^2\to2$ from Equation~\eqref{eq:tight_gamma}, which also gives $\gamma(\BFh)/r\to0$, the ratio satisfies
\begin{equation}
\begin{aligned}
&\frac{\zeta(\BFa)-\zeta(\hat{\BFa})}{(r+4r^2)\sqrt{\gamma(\BFh)/2}+2r\gamma(\BFh)/(2+r)}\\
&\quad=\frac{1+2r}{(1+4r)\sqrt{\gamma(\BFh)/(2r^2)}+2\gamma(\BFh)/\bigl((2+r)r\bigr)}\to1.
\end{aligned}
\label{eq:appn:tsttratio}
\end{equation}

Since every ratio tends to one, none of the bounds examined above can be multiplied by a fixed factor smaller than one uniformly over the class of problems we consider.'''
    source_sha256 = "29a25972c1f7542a766c36f67ee5c7d9f882b76fc5a7380c07d4f7cebae5294f"

    try:
        import sympy as s
    except ImportError:
        message = ("NOT RUN: SymPy is not installed in this Python environment.\n"
                   "No calculation has been marked PASS or FAIL.\n"
                   "Install SymPy, then run this Python file again.\n")
        report_path.write_text(message, encoding="utf-8")
        print(message, end="")
        raise SystemExit(3)

    active_source = remove_tex_comments(source_snapshot)
    if source_path.exists():
        active_source = remove_tex_comments(source_path.read_text(encoding="utf-8"))
    actual_sha256 = hashlib.sha256(active_source.encode("utf-8")).hexdigest()
    if actual_sha256 != source_sha256:
        message = ("NOT RUN: the active source differs from the audited snapshot.\n"
                   "Update the explicitly transcribed assertions before rerunning.\n"
                   "Edits confined to percent comments do not change the snapshot.\n")
        report_path.write_text(message, encoding="utf-8")
        print(message, end="")
        raise SystemExit(3)

    r, u = s.symbols("r u", positive=True)
    x = s.symbols("x", nonnegative=True)
    y1, y2 = s.symbols("y1 y2", positive=True)
    G = s.symbols("G", positive=True)
    theta = s.Integer(1)
    demand = s.Integer(1)
    half = s.Rational(1, 2)
    domain = {r: u / (2 * (1 + u))}  # Bijective parameterization of 0 < r < 1/2.

    incidence = s.eye(2)
    h = s.Matrix([half - r, half + r])
    hhat = s.Matrix([half, half])
    a = incidence * h
    ahat = incidence * hhat
    tau = (s.Integer(1), 1 - r + 2*r*x)
    C = s.Matrix([tau[i].subs(x, a[i]) for i in range(2)])
    Chat = s.Matrix([tau[i].subs(x, ahat[i]) for i in range(2)])

    def entropy(vector):
        return sum(value * s.log(value) for value in vector) / theta

    def beta(vector):
        return sum(s.integrate(tau[i], (x, 0, vector[i])) for i in range(2))

    def logit(costs):
        weights = [s.exp(-cost/theta) for cost in costs]
        return s.Matrix([s.simplify(demand * weight / sum(weights)) for weight in weights])

    L = logit(C)
    Lhat = logit(Chat)
    eta_h = entropy(h)
    eta_L = entropy(L)
    eta_hat = entropy(hhat)
    beta_h = beta(h)
    beta_hat = beta(hhat)

    def psi(vector):
        return beta_h + C.dot(vector-h) + entropy(vector)

    cost_term = s.simplify(C.dot(h-L))
    gamma_raw = psi(h) - psi(L)
    gamma = s.expand(s.expand_log(gamma_raw))
    phi_h = beta_h + eta_h
    phi_hat = beta_hat + eta_hat
    objective_error = s.expand(phi_h-phi_hat)
    h_l2_error = s.sqrt((h-hhat).dot(h-hhat))
    h_l1_error = sum(s.Abs(value) for value in h-hhat)
    link_error = [s.Abs(a[i]-ahat[i]) for i in range(2)]

    audit = Audit(s, r, report_path, active_source, precision_order)
    audit.emit(f"SYMPY {s.__version__}: TIGHTNESS APPENDIX AUDIT")
    audit.emit("Input domain: 0 < r < 1/2; theta = d_w = d_max = 1.")
    audit.emit("All calculations use exact expressions; percent-commented text is excluded.")
    audit.emit("G is a positive placeholder for gamma in algebraic bound substitutions.")
    audit.emit("Every limit substitutes the exact gamma computed from psi, not its claimed expansion.")
    audit.emit("The reference theorem formulas are accepted as quoted inputs and flagged separately.")
    audit.emit(f"Active source SHA256: {actual_sha256}")
    audit.emit()

    # Network identities, feasibility, and equilibrium, active-source lines 3 and 20.
    S_l = [demand, demand]    # W_l = {w} for both links, so S_l = d_w = 1
    audit.equal("line 3", "a_1 = h_1 from the stated incidence matrix", a[0], h[0])
    audit.equal("line 3", "a_2 = h_2 from the stated incidence matrix", a[1], h[1])
    audit.equal("line 3", "d_max = d_w = 1", s.Max(demand), 1)
    for i in range(2):
        audit.equal("line 3", f"S_l = 1 for link {i+1}", S_l[i], 1)
    audit.equal("line 20", "current flows sum to the OD demand", sum(h), demand)
    audit.positive("line 20", "h_1 > 0 for 0 < r < 1/2", h[0], domain)
    audit.positive("line 20", "h_2 > 0 for 0 < r < 1/2", h[1], domain)
    audit.equal("line 20", "equilibrium candidate satisfies OD demand", sum(hhat), demand)
    for i in range(2):
        audit.equal("line 20", f"tau_{i+1}(1/2) = 1", Chat[i], 1)
        audit.equal("line 20", f"equilibrium logit target {i+1} = 1/2", Lhat[i], half)

    audit.equal("eq:appn:errors, line 23", "Euclidean path-flow error = sqrt(2)*r",
                h_l2_error, s.sqrt(2)*r)
    audit.equal("eq:appn:errors, line 23", "OD l1 path-flow error = 2*r", h_l1_error, 2*r)
    for i in range(2):
        audit.equal("eq:appn:errors, line 24", f"absolute link-{i+1} flow error = r", link_error[i], r)

    audit.equal("line 29", "C_1(h) = 1", C[0], 1)
    audit.equal("line 29", "C_2(h) = 1 + 2*r**2", C[1], 1+2*r**2)
    audit.equal("eq:appn:target, line 32", "L_1 = 1/(1 + exp(-2*r**2))", L[0], 1/(1+s.exp(-2*r**2)))
    audit.equal("eq:appn:target, line 33", "L_2 = exp(-2*r**2)/(1 + exp(-2*r**2))",
                L[1], s.exp(-2*r**2)/(1+s.exp(-2*r**2)))
    audit.equal("eq:appn:target, line 33", "L_2 = 1 - L_1", L[1], 1-L[0])
    audit.big_o("line 37", "exp(-2*r**2) = 1 - 2*r**2 + O(r**4)", s.exp(-2*r**2), 1-2*r**2, 4)
    audit.big_o("line 37", "1 + exp(-2*r**2) = 2*(1-r**2) + O(r**4)",
                1+s.exp(-2*r**2), 2*(1-r**2), 4)
    audit.big_o("line 37", "1/(1-r**2) = 1 + r**2 + O(r**4)", 1/(1-r**2), 1+r**2, 4)
    audit.big_o("eq:appn:targetexp, line 40", "L_1 = (1+r**2)/2 + O(r**4)", L[0], (1+r**2)/2, 4)
    audit.equal("eq:appn:targetexp, line 41", "L_2 = 1 - L_1, repeated identity", L[1], 1-L[0])
    audit.big_o("eq:appn:targetexp, line 41", "L_2 = 1/2 - r**2/2 + O(r**4)", L[1], half-r**2/2, 4)

    audit.equal("line 45", "psi_h(h) = beta(h) + eta(h)", psi(h), beta_h+eta_h)
    audit.equal("line 45", "psi_h(L) = beta(h) + C(h).(L-h) + eta(L)",
                psi(L), beta_h+C.dot(L-h)+eta_L)
    audit.equal("eq:appn:gammadef, line 48", "exact gamma representation equals psi_h(h)-psi_h(L)", gamma, gamma_raw)
    audit.equal("eq:appn:gammadef, line 49", "beta cancels: gamma = C(h).(h-L) + eta(h)-eta(L)",
                gamma_raw, cost_term+eta_h-eta_L)
    audit.big_o("eq:appn:diff, line 56", "h_1-L_1 = (1/2-r)-(1/2+r**2/2) + O(r**4)",
                h[0]-L[0], (half-r)-(half+r**2/2), 4)
    audit.equal("eq:appn:diff, line 57", "(1/2-r)-(1/2+r**2/2) = -r-r**2/2",
                (half-r)-(half+r**2/2), -r-r**2/2)
    audit.big_o("eq:appn:diff, line 58", "h_2-L_2 = (1/2+r)-(1/2-r**2/2) + O(r**4)",
                h[1]-L[1], (half+r)-(half-r**2/2), 4)
    audit.equal("eq:appn:diff, line 59", "(1/2+r)-(1/2-r**2/2) = r+r**2/2",
                (half+r)-(half-r**2/2), r+r**2/2)
    audit.big_o("eq:appn:ch, line 65", "C(h).(h-L) = 2*r**3 + O(r**4)", cost_term, 2*r**3, 4)
    audit.equal("eq:appn:nh, line 71", "eta(h) = h_1*log(h_1) + h_2*log(h_2)",
                eta_h, h[0]*s.log(h[0])+h[1]*s.log(h[1]))
    audit.big_o("eq:appn:nh, line 80", "eta(h) = -log(2) + 2*r**2 + O(r**3)", eta_h, -s.log(2)+2*r**2, 3)
    audit.equal("eq:appn:nlh, lines 86-87", "eta(L) = L_1*log(L_1) + L_2*log(L_2)",
                eta_L, L[0]*s.log(L[0])+L[1]*s.log(L[1]))
    audit.big_o("eq:appn:nlh, line 97", "eta(L) = -log(2) + O(r**3)", eta_L, -s.log(2), 3)
    audit.big_o("eq:tight_gamma, line 104", "gamma = 2*r**3 + (-log(2)+2*r**2) - (-log(2)) + O(r**3)",
                gamma, 2*r**3+(-s.log(2)+2*r**2)-(-s.log(2)), 3)
    audit.big_o("eq:tight_gamma, line 105", "gamma = 2*r**2 + O(r**3)", gamma, 2*r**2, 3)
    audit.tends("line 109", "gamma/r**2 -> 2", gamma/r**2, 2)

    audit.equal("eq:appn:objdecomp, lines 115-116", "objective gap splits into beta gap and entropy gap",
                objective_error, (beta_h-beta_hat)+(eta_h-eta_hat))
    beta_generic = beta(s.Matrix([y1, y2]))
    audit.equal("eq:appn:beta, lines 123-124", "the two integrals give h_1+(1-r)*h_2+r*h_2**2",
                beta_generic, y1+(1-r)*y2+r*y2**2)
    audit.equal("eq:appn:betavals, line 131", "beta(h) = 1-r/4+r**3", beta_h, 1-r/4+r**3)
    audit.equal("eq:appn:betavals, line 132", "beta(hhat) = 1/2+(1-r)/2+r/4", beta_hat, half+(1-r)/2+r/4)
    audit.equal("eq:appn:betavals, line 132", "1/2+(1-r)/2+r/4 = 1-r/4", half+(1-r)/2+r/4, 1-r/4)
    audit.equal("line 136", "beta(h)-beta(hhat) = r**3", beta_h-beta_hat, r**3)
    audit.equal("line 136", "eta(hhat) = (1/2)*log(1/2)+(1/2)*log(1/2)", eta_hat, 2*half*s.log(half))
    audit.equal("line 136", "eta(hhat) = -log(2)", eta_hat, -s.log(2))
    audit.big_o("line 136", "eta(h) = -log(2)+2*r**2+O(r**3), repeated expansion", eta_h, -s.log(2)+2*r**2, 3)
    audit.big_o("line 136", "eta(h)-eta(hhat) = 2*r**2+O(r**3)", eta_h-eta_hat, 2*r**2, 3)
    audit.big_o("eq:appn:objerr, line 138", "phi(h)-phi(hhat) = r**3+2*r**2+O(r**3)",
                objective_error, r**3+2*r**2, 3)
    audit.big_o("eq:appn:objerr, line 138", "phi(h)-phi(hhat) = 2*r**2+O(r**3)", objective_error, 2*r**2, 3)
    audit.big_o("eq:appn:objratio, line 143", "objective-ratio numerator is 2*r**2+O(r**3)", objective_error, 2*r**2, 3)
    audit.big_o("eq:appn:objratio, line 143", "objective-ratio denominator is 2*r**2+O(r**3)", gamma, 2*r**2, 3)
    audit.tends("eq:appn:objratio, line 143", "objective error / gamma -> 1", objective_error/gamma, 1)

    dmax = s.Max(demand)
    mh_G = s.sqrt(theta*dmax*G)
    mod_G = s.sqrt(2*theta*demand*G)
    path_ratio_G = s.sqrt(2)*r/s.sqrt(G)
    audit.equal("line 148", "m_h = sqrt(theta*d_max*G) = sqrt(G)", mh_G, s.sqrt(G))
    audit.equal("line 148", "OD l1 bound = sqrt(2*theta*d_w*G) = sqrt(2*G)", mod_G, s.sqrt(2*G))
    audit.tends("line 148", "gamma/r**2 -> 2, repeated prerequisite", gamma/r**2, 2)
    audit.equal("eq:appn:pathratio, line 151", "l2 error / m_h = sqrt(2)*r/sqrt(G)", h_l2_error/mh_G, path_ratio_G)
    audit.tends("eq:appn:pathratio, line 151", "l2 error / m_h -> 1", (h_l2_error/mh_G).subs(G, gamma), 1)
    audit.equal("eq:appn:pathratio, line 152", "l1 error / sqrt(2*G) = sqrt(2)*r/sqrt(G)", h_l1_error/mod_G, path_ratio_G)
    audit.tends("eq:appn:pathratio, line 152", "l1 error / OD bound -> 1", (h_l1_error/mod_G).subs(G, gamma), 1)

    # Link flow bound: tilde m_a^l (Lemma lem:3_linkbound), tightened with R_a^l (Theorem thm:3_linkbound).
    # tau_l' does not depend on x for either link, so its minimum (R_a^l) and maximum (L_a^l)
    # over any flow interval both equal that constant.
    slopes = [s.diff(cost, x) for cost in tau]
    audit.equal("line 157", "tau_1' does not depend on the flow", s.diff(slopes[0], x), 0)
    audit.equal("line 157", "tau_2' does not depend on the flow", s.diff(slopes[1], x), 0)
    R = slopes
    audit.equal("line 157", "R_a^1 = 0", R[0], 0)
    audit.equal("line 157", "R_a^2 = 2*r", R[1], 2*r)
    ma_tilde_G = [s.sqrt(theta*S_l[i]*G/2) for i in range(2)]
    ma_G = [ma_tilde_G[i]/s.sqrt(1 + theta*S_l[i]*R[i]/4) for i in range(2)]
    for i in range(2):
        audit.equal("line 157", f"tilde m_a^{i+1} = sqrt(theta*S_l*G/2) = sqrt(G/2)", ma_tilde_G[i], s.sqrt(G/2))
    audit.equal("line 157", "m_a^1 = tilde m_a^1 = sqrt(G/2)", ma_G[0], s.sqrt(G/2))
    audit.equal("line 157", "m_a^2 = tilde m_a^2/sqrt(1+r/2) = sqrt(G/(2+r))", ma_G[1], s.sqrt(G/(2+r)))
    link1_ratio_G = s.sqrt(2)*r/s.sqrt(G)
    link2_ratio_G = s.sqrt(2+r)*r/s.sqrt(G)
    audit.equal("eq:appn:linkratio, line 159", "link-1 flow ratio = sqrt(2)*r/sqrt(G)", link_error[0]/ma_G[0], link1_ratio_G)
    audit.tends("eq:appn:linkratio, line 159", "link-1 flow ratio -> 1", (link_error[0]/ma_G[0]).subs(G, gamma), 1)
    audit.equal("eq:appn:linkratio, line 161", "link-2 flow ratio = sqrt(2+r)*r/sqrt(G)", link_error[1]/ma_G[1], link2_ratio_G)
    audit.tends("eq:appn:linkratio, line 161", "link-2 flow ratio -> 1", (link_error[1]/ma_G[1]).subs(G, gamma), 1)

    cost_errors = [s.Abs(C[i]-Chat[i]) for i in range(2)]
    audit.equal("line 165", "link 1 has zero cost error", cost_errors[0], 0)
    audit.equal("line 165", "link 1 has zero cost bound", slopes[0]*ma_G[0], 0)
    audit.equal("line 165", "link-2 cost error = 2*r*abs(a_2-ahat_2)", cost_errors[1], 2*r*link_error[1])
    audit.equal("line 165", "link-2 cost error = 2*r**2", cost_errors[1], 2*r**2)
    audit.equal("line 165", "L_a^2 = 2*r (tau_2' is constant)", slopes[1], 2*r)
    audit.equal("line 165", "m_a^2 = sqrt(G/(2+r)), repeated from the link flow bound", ma_G[1], s.sqrt(G/(2+r)))
    cost_ratio_G = cost_errors[1]/(slopes[1]*ma_G[1])
    audit.equal("eq:appn:costratio, line 168", "link-2 cost ratio = 2*r**2/(2*r*sqrt(G/(2+r)))",
                cost_ratio_G, 2*r**2/(2*r*s.sqrt(G/(2+r))))
    audit.equal("eq:appn:costratio, line 169", "link-2 cost ratio = sqrt(2+r)*r/sqrt(G)", cost_ratio_G, link2_ratio_G)
    audit.tends("eq:appn:costratio, line 169", "link-2 cost ratio -> 1", cost_ratio_G.subs(G, gamma), 1)

    generic_flows = s.Matrix([y1, y2])
    zeta_generic = sum(generic_flows[i]*tau[i].subs(x, generic_flows[i]) for i in range(2))
    zeta_h = zeta_generic.subs({y1: a[0], y2: a[1]}, simultaneous=True)
    zeta_hat = zeta_generic.subs({y1: ahat[0], y2: ahat[1]}, simultaneous=True)
    tstt_error = s.expand(zeta_h-zeta_hat)
    audit.equal("line 174", "TSTT = a_1+(1-r)*a_2+2*r*a_2**2", zeta_generic, y1+(1-r)*y2+2*r*y2**2)
    audit.equal("line 174", "equilibrium TSTT = 1", zeta_hat, 1)
    audit.equal("line 174", "current TSTT = 1+r**2+2*r**3", zeta_h, 1+r**2+2*r**3)
    audit.equal("line 174", "TSTT error = r**2+2*r**3", tstt_error, r**2+2*r**3)
    q = [s.simplify(s.diff(zeta_generic, generic_flows[i]).subs({y1: a[0], y2: a[1]}, simultaneous=True)) for i in range(2)]
    audit.equal("line 174", "q_1 = 1", q[0], 1)
    audit.equal("line 174", "q_2 = tau_2(a_2)+a_2*tau_2'(a_2)", q[1], C[1]+a[1]*slopes[1].subs(x, a[1]))
    audit.equal("line 174", "q_2 = 1+r+4*r**2", q[1], 1+r+4*r**2)
    audit.positive("line 174", "q_2 > q_1, so the spread is q_2-q_1", q[1]-q[0], domain)
    Q = s.sqrt(demand)*(s.Max(*q)-s.Min(*q))
    audit.equal("line 174", "Q_a = sqrt(d_w)*(q_2-q_1)", Q, s.sqrt(demand)*(q[1]-q[0]))
    audit.equal("line 174", "Q_a = r+4*r**2", Q, r+4*r**2)
    curvature = [2*s.diff(cost, x)+x*s.diff(cost, x, 2) for cost in tau]
    for i in range(2):
        audit.equal("line 174", f"second derivative of a_{i+1}*tau_{i+1}(a_{i+1}) equals 2*tau'+x*tau''",
                    s.diff(x*tau[i], x, 2), curvature[i])
        audit.equal("line 174", f"2*tau_{i+1}'(x)+x*tau_{i+1}''(x) does not depend on the flow", s.diff(curvature[i], x), 0)
    audit.equal("line 174", "2*tau_1'(x)+x*tau_1''(x) = 0", curvature[0], 0)
    audit.equal("line 174", "2*tau_2'(x)+x*tau_2''(x) = 4*r", curvature[1], 4*r)
    K = [s.Abs(value) for value in curvature]
    audit.equal("line 174", "K_a^1 = 0", K[0], 0)
    audit.equal("line 174", "K_a^2 = 4*r", K[1], 4*r)
    audit.equal("line 174", "(m_a^2)**2 = G/(2+r)", ma_G[1]**2, G/(2+r))
    bound_full_G = Q*s.sqrt(theta*G/2)+sum(K[i]*ma_G[i]**2/2 for i in range(2))
    bound_no_zero_G = Q*s.sqrt(theta*G/2)+K[1]*ma_G[1]**2/2
    bound_claim_G = (r+4*r**2)*s.sqrt(G/2)+2*r*G/(2+r)
    audit.equal("eq:appn:tsttbound, line 177", "the link-1 second-order term vanishes", bound_full_G, bound_no_zero_G)
    audit.equal("eq:appn:tsttbound, line 178", "Q_a*sqrt(theta*G/2) = (r+4*r**2)*sqrt(G/2)", Q*s.sqrt(theta*G/2), (r+4*r**2)*s.sqrt(G/2))
    audit.equal("eq:appn:tsttbound, line 178", "(1/2)*K_a^2*(m_a^2)**2 = 2*r*G/(2+r)", K[1]*ma_G[1]**2/2, 2*r*G/(2+r))
    audit.equal("eq:appn:tsttbound, line 178", "full TSTT bound = (r+4*r**2)*sqrt(G/2)+2*r*G/(2+r)", bound_full_G, bound_claim_G)
    audit.tends("line 182", "gamma/r**2 -> 2, TSTT prerequisite", gamma/r**2, 2)
    audit.tends("line 182", "gamma/r -> 0", gamma/r, 0)
    audit.equal("eq:appn:tsttratio, line 186", "TSTT error divided by r**2 = 1+2*r", tstt_error/r**2, 1+2*r)
    scaled_denominator_G = (1+4*r)*s.sqrt(G/(2*r**2))+2*G/((2+r)*r)
    audit.equal("eq:appn:tsttratio, line 186", "TSTT bound divided by r**2 has the stated denominator", bound_full_G/r**2, scaled_denominator_G)
    audit.equal("eq:appn:tsttratio, line 185", "the stated TSTT ratio equals error / bound",
                tstt_error/bound_claim_G, tstt_error/bound_full_G)
    audit.equal("eq:appn:tsttratio, lines 185-186", "dividing numerator and denominator by r**2 preserves the TSTT ratio", tstt_error/bound_full_G, (1+2*r)/scaled_denominator_G)
    audit.tends("eq:appn:tsttratio, line 186", "scaled TSTT numerator -> 1", 1+2*r, 1)
    audit.tends("eq:appn:tsttratio, line 186", "scaled TSTT denominator -> 1", scaled_denominator_G.subs(G, gamma), 1)
    audit.tends("eq:appn:tsttratio, line 186", "exact TSTT error / exact quoted bound -> 1", (tstt_error/bound_full_G).subs(G, gamma), 1)

    factor = u/(1+u)
    epsilon = (1-factor)/2
    audit.positive("line 191", "for every 0 < factor < 1, epsilon=(1-factor)/2 > 0", epsilon, {})
    audit.equal("line 191", "1-epsilon exceeds factor by epsilon", 1-epsilon-factor, epsilon)

    audit.missing("eq:2_suepathflow; eq:3_fixedcostobjective; eq:2_sueobjective",
                  "Quoted definitions agree with the missing main-paper equations",
                  "The attachment supplies their specialized formulas, but not the referenced equations. Their substitutions are checked above.")
    audit.missing("thm:3_objectivegap; thm:3_globalpathbound; lem:3_linkbound; thm:3_linkbound; cor:3_linkcostbound; thm:3_tsttbound",
                  "Quoted bound formulas and constants agree with the missing theorems",
                  "The theorem statements are absent. This script checks the quoted formulas' algebra and limits, not their general proofs.")
    audit.missing("lines 3 and 191",
                  "This network family satisfies all assumptions of the full problem class",
                  "The assumptions are absent. The limiting fixed-factor conclusion is conditional on applicability of the quoted bounds.")

    result = audit.finish({
        "L_1": L[0], "L_2": L[1],
        "C(h).(h-L)": cost_term,
        "eta(h)": eta_h, "eta(L)": eta_L,
        "gamma(h)": gamma,
        "phi(h)-phi(hhat)": objective_error,
        "TSTT bound": bound_full_G.subs(G, gamma),
    })
    raise SystemExit(result)