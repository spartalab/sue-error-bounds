# Error Bounds for Logit-based Stochastic User Equilibrium Traffic Assignment

Code for the paper *Error Bounds for Logit-based Stochastic User Equilibrium Traffic Assignment* by Debojjal Bagchi and Stephen D. Boyles.

For any feasible path flow of a logit SUE with fixed path sets, the code computes upper bounds on the errors in the SUE objective, path flows, link flows, link costs, and total system travel time (TSTT) relative to the equilibrium. It reproduces every table and figure of the computational results section and the appendix section of the paper.

---

## Repository structure

```text
.
├── main.py                  # Tightness results (Section 5.1 and Appendix): runs, tables and figures
├── capacityDesign.py        # Network design case study (Section 5.2)
├── SUEPath.py               # Path-based logit SUE (FP-BB1) and the error bounds
├── genPaperPlots.py         # Paper tables (table.tex) and figures, called by main.py
├── genPlots.py              # All figures of every bound run, called by main.py
├── readNetwork.py           # Reads the network and demand .dat files
├── helper.py                # Converts TNTP files to .dat files
├── appendix/
│   ├── tightness.py             # Symbolic (SymPy) check of the tightness appendix
│   └── precision.py             # Checks long double support and speed on this machine
├── pathGenerator/
│   ├── pathSetGenerator_yen.py  # 20 shortest acyclic paths per OD pair (Yen's algorithm)
│   ├── fix_pikles.py            # Converts node IDs in the path files to strings
│   └── analyzePaths.py          # Path set diversity: CV and Jaccard columns of Table tab:4_res
└── testNetworks/            # Sioux Falls, Anaheim, Winnipeg, Chicago Sketch (TNTP format)
```

---

## Reproducing the paper results


**1. Path sets** (written to `Paths/`)

    python pathGenerator/pathSetGenerator_yen.py
    python pathGenerator/fix_pikles.py
    python pathGenerator/analyzePaths.py

Set the networks to generate in `first_thru_node_dict` of `pathSetGenerator_yen.py`.

**2. Tightness of the bounds**

    python main.py

For each network and theta in {0.5, 1.5}, this solves a reference equilibrium to RGAP 1e-12 (`optimalSolutions/`, skipped if it already exists), then reruns from the same start to RGAP 1e-8 while recording all bounds and true errors (`boundResults/<network>/<run>/plots.csv`). It then writes the paper tables and figures to `paperResults/` (`table.tex` and PDFs) and the figures of every run to `plots/`.

**3. Network design case study**

    python capacityDesign.py

Runs the local search on Chicago Sketch once without and once with the bounds. Results are saved in `capDesRes/` (one CSV per run, plus a summary file with every number reported in the paper), and the two figures to `paperResults/`. A run whose CSV already exists is not repeated. Delete the CSVs to rerun.

**4. Checks**

    python appendix/tightness.py     # writes appendix/tightness_sympy_results.txt
    python appendix/precision.py     # long double precision and speed

`tightness.py` verifies every equation of the tightness appendix of the paper symbolically. `tightness.py` contains a copy of the appendix and checks every equation in it with SymPy.

All computations use long double (`np.longdouble`). `precision.py` shows what the current machine supports.

---

## Requirements

    pip install numpy scipy pandas matplotlib networkx tqdm sympy

Tested with Python 3.12.5. LaTeX is optional: if it is installed, figures use LaTeX fonts.

---

## Acknowledgements

- Networks from [TransportationNetworks](https://github.com/bstabler/TransportationNetworks).
- The SUE code was adapted from [Traffic-Assignment](https://github.com/prameshk/Traffic-Assignment).

## Contact

Debojjal Bagchi, debojjalb@utexas.edu, https://debojjalb.github.io