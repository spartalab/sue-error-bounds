"""
Regerates all the tightness resutls for the paper.
"""
import json
import os
import shutil
from pathlib import Path

from readNetwork import Network
from SUEPath import StochasticUE
from helper import convert_network_tntp_to_dat, convert_demand_tntp_to_dat

NETWORKS = ["Sioux Falls", "Anaheim", "Winnipeg", "Chicago Sketch"]
THETAS = [0.5, 1.5]
K = 20                  # paths per OD pair (Yen's k-shortest paths)

# Passed to every assignment call. Each call stops at its accuracy or after time_limit seconds.
OPTIONS = {"stepRule": "BB1", "initType": "logitFreeFlow", "I_s": 10,
           "demand_multiplier": 1.0, "time_limit": 60*60, "save_log": False}

os.chdir(Path(__file__).resolve().parent)

for networkName in NETWORKS:
    folder = "testNetworks/" + networkName + "/"
    if not os.path.exists(folder + "network.dat"):
        convert_network_tntp_to_dat(folder + "net.tntp", folder + "network.dat")
    if not os.path.exists(folder + "demand.dat"):
        convert_demand_tntp_to_dat(folder + "trips.tntp", folder + "demand.dat")

    for theta in THETAS:
        print("Network:", networkName, "theta:", theta)
        network = Network()
        network.readNetwork(folder)
        network.readDemand(folder)
        solver = StochasticUE(network, networkName, theta, False, K)

        # File names written by SUEPath
        solutionFile = f"optimalSolutions/{networkName}_theta{theta}_d1.0x_K{K}_BB1_solution.pkl"
        resultsFolder = f"boundResults/{networkName}/theta{theta}_d1.0x_K{K}_BB1/"

        # 1. Reference equilibrium hat h
        if not os.path.exists(solutionFile):
            solver.assignment(accuracy=1e-12, maxIter=10**9, networkName=networkName,
                              save_final_solution=True, **OPTIONS)
            if not os.path.exists(solutionFile):
                raise RuntimeError("RGAP 1e-12 was not reached; increase time_limit.")

        # 2. Bound run (always run again; plots.csv is overwritten)
        solver.assignment(accuracy=1e-8, maxIter=10**9, networkName=networkName,
                          calcBounds=True, **OPTIONS)

        # 3. Network statistics for Table tab:4_res
        nodes = None
        for line in open(folder + "net.tntp"):
            if line.startswith("<NUMBER OF NODES>"):
                nodes = int(line.split()[3])
        summary = {"network": networkName, "od_pairs": len(solver._od_order),
                   "links": len(solver.linkSet), "nodes": nodes,
                   "demand": float(solver.d_w.sum()), "paths": int(solver.totalPaths)}
        Path(resultsFolder + "summary.json").write_text(json.dumps(summary, indent=2))

# 4. Figures and tables, drawn from scratch
import matplotlib.pyplot as plt

shutil.rmtree("plots", ignore_errors=True)
import genPlots
genPlots.makeRunPlots(resultsFolder="boundResults", plotsFolder="plots", boundAccuracy=1e-8)

plt.rcdefaults()                  # genPaperPlots sets its own matplotlib style when imported
import genPaperPlots
genPaperPlots.makePaperPlots()

print("Done.")