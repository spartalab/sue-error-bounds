"""
SUEPath.py
Implements the Stochastic User Equilibrium (SUE) model using a Logit formulation
Adds the error bounds from the paper. 
"""
import time
import pickle
import numpy as np
from collections import deque
import os
import csv

from scipy.sparse import csr_matrix
FLOAT = np.longdouble

class StochasticUE:
    """
    Stochastic User Equilibrium (SUE) model
    """
    def __init__(self, network, networkName, theta, debug, num_paths, trackHistory=False):
        self.networkName = networkName
        self.tripSet = network.tripSet
        self.zoneSet = network.zoneSet
        self.linkSet = network.linkSet
        self.nodeSet = network.nodeSet
        self.originZones = set([k[0] for k in self.tripSet])

        self.linkFlows = {l: 0.0 for l in self.linkSet}
        self.linkCosts = {l: 0.0 for l in self.linkSet}
        self.linkCostDerivatives = {l: 0.0 for l in self.linkSet}

        self.theta = theta
        self.k = num_paths
        self.debug = debug
        self.tstt = None
        self.gap = 1e7
        self.rtol = 0.01
        self.alpha = 1.0

        self.allPathsODCache = self._allPathOD()
        
        self.optimalSol = None

        # Caching & Indexing Setup
        print("Preprocessing network and paths...")
        self._createGlobalPathIndex()
        self._createFullLinkPathIncidence()
        self._precomputeODVectors()

        print(f"Initialization complete. Network: {networkName}, Theta: {theta}, k: {num_paths}")
        
        self.h = np.zeros(self.totalPaths, dtype=FLOAT)   # Current Path Flows
        self.Lh = np.zeros(self.totalPaths, dtype=FLOAT)  # Target Path Flows
        self.P = np.zeros(self.totalPaths, dtype=FLOAT)   # Logit Probabilities
        self.C = np.zeros(self.totalPaths, dtype=FLOAT)   # Path Costs

        self.h_n_minus_one = None
        self.h_n_minus_two = None
        self.lh_n_minus_one = None
        self.lh_n_minus_two = None
        
        self.it = 0
        self.prev_gap = deque(maxlen=3)

        self.trackHistory = trackHistory
        if trackHistory:
            self.history = {
                'Iteration': [],
                'h': [],
                'Lh': [],
                'gap': [], # Note: the gaps are offset by 1 iter from h since init is 10^7
                'alpha': []
            }
        else:
            self.history = None


        ####################################################
        # Shorthands from the paper (Tables 1 and 2), used by the bound section.
        ####################################################

        # d_w: demand of each OD pair, in the same order as self._od_order
        self.d_w = np.array([self.tripSet[w].demand for w in self._od_order],  dtype=FLOAT)

        # d_max = max_w d_w
        self.d_max = np.max(self.d_w)

        # OD pairs with positive demand but no cached paths are skipped and never assigned
        self.totalODs = sum(1 for trip in self.tripSet.values() if trip.demand > 0)
        if self.totalODs != len(self._od_order):
            missing_ODs = [OD for OD in self.tripSet if self.tripSet[OD].demand > 0 and OD not in self._od_order]
            for o, d in missing_ODs:
                if o != d:
                    raise ValueError(f"Mismatch in total OD pairs: {self.totalODs} vs {len(self._od_order)}. Check path generation.")


        self.totalDemand = sum(trip.demand for trip in self.tripSet.values() if trip.demand > 0)
        self.assignedDemand = np.sum(self.d_w)
        print(f"Assigned OD pairs: {len(self._od_order)} of {self.totalODs}, demand: {self.assignedDemand} of {self.totalDemand}")

        # p_l: BPR power of each link, in the same order as the rows of D
        self.p_l = np.array([self.linkSet[link].beta for link in self.linkIndexMap], dtype=FLOAT)
        # NOTE FOR SELF: (add this to readme); for all evaluatiosn the beta is needed to be greater than 1 or exactly 0. 

        # W_l: OD pairs with at least one path using link l and at least one path not using it
        self.W_l = self.computeWl()

        # S_l = sum of d_w over w in W_l
        self.S_l = self.computeSl()

        # Bound run state (set in assignment when calcBounds=True)
        self.boundsReference = None       # hat h, hat a, T(hat a), phi(hat h), zeta(hat a), loaded from optimalSolutions/
        self.boundsStream = None          # open plots.csv file
        self.boundsWriter = None          # csv writer for plots.csv
        self.boundsClock = None           # time of the previous recordBounds call


    def _allPathOD(self):
        """Loads precomputed k-shortest paths."""
        safename = self.networkName.replace(" Network", "").replace(" ", "_")
        file_path = f'Paths/allPathsODCache_{self.k}_{safename}.pkl'
        
        print(f"Attempting to load precomputed paths from {file_path}...")
        try:
            with open(file_path, 'rb') as f:
                print('Loading all paths for each OD from pickle')
                raw = pickle.load(f)
                allPathOD = {(str(o), str(d)): paths for (o, d), paths in raw.items()}

                print("Successfully loaded precomputed paths. Developing final path cache dictionary...")
                to_delete = [OD for OD in allPathOD if OD not in self.tripSet or self.tripSet[OD].demand <= 0]
                for OD in to_delete:
                    if OD in allPathOD:
                        del allPathOD[OD]
                return allPathOD
        except Exception as e:
            print("Pleased generate path cache by running pathSetGenerator_<version>.py before running the main assignment.")
            print("If you already generated the paths, please check that pathGenerator/fix_pikles.py was run")
            print("Netowk Name:", self.networkName)
            raise e
           
        return allPathOD

    def _createGlobalPathIndex(self):
        """
        Build a flat, global ordering of all paths across all OD pairs.

        Iterates over ODs in sorted (origin, destination) order, skipping any OD
        that was dropped from allPathsODCache due to zero demand. Within each OD,
        paths are appended in the order they appear in the cache. The position of
        each path in this flat ordering is its global index, used everywhere as
        the column index of the link-path incidence matrix and the position in
        the flat arrays self.h, self.Lh, self.P, self.C.

        Updates
        -------
        self.globalPathList : list of (OD, path_str)
            Flat list of length self.totalPaths.
        self.globalPathIndex : dict (OD, path_str) -> int
            Reverse lookup from path identifier to global index.
        self.odBlockSlices : dict OD -> slice
            For each OD, the contiguous slice into the flat arrays holding that
            OD's paths.
        self.totalPaths : int
            Total number of paths across all surviving ODs.

        Example
        -------
        Suppose sorted ODs are A, B, C with 2, 3, 2 paths respectively. Then:
            globalPathList   = [(A,p1), (A,p2), (B,p1), (B,p2), (B,p3), (C,p1), (C,p2)]
            globalPathIndex  = {(A,p1):0, (A,p2):1, (B,p1):2, ..., (C,p2):6}
            odBlockSlices    = {A: slice(0,2), B: slice(2,5), C: slice(5,7)}
            totalPaths       = 7
        """
        self.globalPathList = []
        self.globalPathIndex = {}
        self.odBlockSlices = {}
        idx = 0
        
        for orgn in sorted(self.originZones):
            for dest in sorted(self.zoneSet[orgn].destList):
                OD = (orgn, dest)
                if OD not in self.allPathsODCache:
                    continue

                start = idx
                for path in self.allPathsODCache[OD]:
                    self.globalPathList.append((OD, str(path)))
                    self.globalPathIndex[(OD, str(path))] = idx
                    idx += 1
                self.odBlockSlices[OD] = slice(start, idx)
                
        self.totalPaths = idx
        print(f"Total paths across all OD pairs: {self.totalPaths}")

    def _createFullLinkPathIncidence(self):
        """
        Build the sparse link-path incidence matrix D.

        Constructs D as a sparse matrix of shape (numLinks, totalPaths), where
        D[l, p] = 1 if link l lies on path p and 0 otherwise. Links are indexed
        by the order returned by self.linkSet.keys(), and paths are indexed by
        their position in self.globalPathList. For each global path index, the
        corresponding path sequence is looked up in self.allPathsODCache, its
        consecutive node pairs are converted to links, and each link contributes
        a unit entry in the appropriate column of D. The transpose is also
        cached for fast path-to-link aggregation.

        Updates
        -------
        self.linkIndexMap : dict link -> int
            Maps each link (i, j) to its row index in D.
        self.fullLinkPathIncidence : scipy.sparse.csr_matrix
            The link-path incidence matrix D of shape (numLinks, totalPaths).
        self.fullLinkPathIncidence_T : scipy.sparse.csr_matrix
            The transpose D^T of shape (totalPaths, numLinks).
        """
        links = list(self.linkSet.keys())
        self.linkIndexMap = {l: i for i, l in enumerate(links)}
        
        rows, cols, data = [], [], []
        for gidx, (OD, path_str) in enumerate(self.globalPathList):
            for path in self.allPathsODCache[OD]:
                if str(path) == path_str:
                    pathLinks = [(path[j], path[j+1]) for j in range(len(path)-1)]
                    for link in pathLinks:
                        rows.append(self.linkIndexMap[link])
                        cols.append(gidx)
                        data.append(1.0)
                    break

        print("Constructing full link-path incidence matrix...")
        self.fullLinkPathIncidence = csr_matrix((data, (rows, cols)), shape=(len(links), self.totalPaths), dtype=FLOAT)
        self.fullLinkPathIncidence_T = csr_matrix((data, (cols, rows)), shape=(self.totalPaths, len(links)), dtype=FLOAT)
        print("Dimensions of full link-path incidence matrix: ", self.fullLinkPathIncidence.shape)

    def _precomputeODVectors(self):
        """
        Precompute OD ordering, split indices, and flat demand arrays for
        vectorized operations.

        Establishes a OD ordering by iterating over origins and
        destinations in sorted order and skipping any OD absent from
        allPathsODCache. From this ordering, builds the per-OD path counts and
        two arrays of split indices used to slice or segment the flat
        path-level arrays (h, Lh, P, C) into per-OD blocks. Also constructs a
        flat array of length totalPaths in which each entry holds the demand
        of the OD that owns the corresponding path. 

        Updates
        -------
        self._od_order : list of (origin, destination)
            Canonical OD ordering matching the path block layout in the flat
            arrays. Length equals the number of surviving ODs.
        self._path_counts : list of int
            Number of paths per OD, aligned with self._od_order.
        self._od_split_indices : np.ndarray
            Cumulative path counts with the final entry dropped, suitable as
            the indices argument to np.split for partitioning a flat
            path-level array into per-OD blocks.
        self._od_split_indices_0 : np.ndarray
            Same as _od_split_indices but with a leading 0 prepended, giving
            the start index of each OD's block. Length equals the number of
            ODs.
        self._path_demands : np.ndarray
            Flat array of length self.totalPaths where entry g holds the
            demand of the OD that owns global path index g.

        Raises
        ------
        ValueError
            If any OD's demand is not a Python float or numpy floating type.

        Example
        -------
        Suppose surviving ODs in sorted order are A, B, C with 2, 3, 2 paths
        respectively, and demands 10.0, 20.0, 30.0. Then:
            _od_order            = [A, B, C]
            _path_counts         = [2, 3, 2]
            _od_split_indices    = [2, 5]
            _od_split_indices_0  = [0, 2, 5]
            _path_demands        = [10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0]
        """
        print("Precomputing OD vectors for vectorized operations...")
        self._od_order = []
        for orgn in sorted(self.originZones):
            for dest in sorted(self.zoneSet[orgn].destList):
                OD = (orgn, dest)
                
                if OD in self.allPathsODCache:
                    self._od_order.append(OD)
                    
        self._path_counts = [len(self.allPathsODCache[OD]) for OD in self._od_order]
        self._od_split_indices = np.cumsum(self._path_counts)[:-1]
        self._od_split_indices_0 = np.insert(self._od_split_indices, 0, 0)

        for OD in self._od_order:
            demand = self.tripSet[OD].demand
            if not isinstance(demand, (float, np.floating)):
                raise ValueError(f"Demand for OD {OD} is not a float type: {demand} (type: {type(demand)})")
                
        self._path_demands = np.repeat(
            np.array([self.tripSet[OD].demand for OD in self._od_order], dtype=FLOAT), 
            self._path_counts
        )

    def _linkCost(self, link, flow):
        """Computes the cost of a link given its flow using the BPR function."""
        if not self.debug:
            l = self.linkSet[link]
            return l.fft * (1 + l.alpha * (flow / l.capacity) ** l.beta) + l.tollInTime

        else:
            costs = {('1', '2'): 10 * flow, ('1', '3'): 50 + flow, 
                     ('2', '3'): 10 + flow, ('2', '4'): 50 + flow, ('3', '4'): 10 * flow}
            return costs.get(link, 0)


    def _linkCostDerivative(self, link, flow):
        """Computes the derivative of the link cost with respect to flow."""
        if not self.debug:
            l = self.linkSet[link]
            if l.alpha == 0 or l.beta == 0: # Winnipeg constant travel times have alpha = 0
                return 0.0 # This is still consistent w paper; the derivative of a constant cost is zero
            return l.fft * l.alpha * l.beta * (flow / l.capacity) ** (l.beta - 1) / l.capacity

    def _updateLinkFlows(self, h_array):
        """Updates link flows using a provided path flow array."""
        v = self.fullLinkPathIncidence @ h_array
        for link, idx in self.linkIndexMap.items():
            self.linkFlows[link] = v[idx]

    def _updateLinkCostsAndDerivatives(self):
        """Batch updates link costs and derivatives."""
        for link in self.linkSet:
            flow = self.linkFlows[link]
            self.linkCosts[link] = self._linkCost(link, flow)
            self.linkCostDerivatives[link] = self._linkCostDerivative(link, flow)

    def updatePathCosts(self, h_array):
        """Updates costs natively as 1D arrays."""
        self._updateLinkFlows(h_array)
        self._updateLinkCostsAndDerivatives()

        link_costs = np.array([self.linkCosts[link] for link in self.linkIndexMap], dtype=FLOAT)
        link_flows = np.array([self.linkFlows[link] for link in self.linkIndexMap], dtype=FLOAT)

        self.C = self.fullLinkPathIncidence_T @ link_costs
        self.tstt = np.dot(link_costs, link_flows)

    def updateLogitAndTargetFlows(self):
        """
        Compute logit path choice probabilities and target path flows using
        fully vectorized operations on the flat path-level arrays.

        Note (very important): For this to work every OD shoudld have atleast 
        one path.

        Updates
        -------
        self.P : np.ndarray
            Flat array of length totalPaths holding logit choice probabilities
            within each OD block.
        self.Lh : np.ndarray
            Flat array of length totalPaths holding target path flows
            (probability times OD demand).
        """

        V = -self.theta * self.C
        
        max_V_per_OD = np.maximum.reduceat(V, self._od_split_indices_0) # This gives the max V for each OD, aligned with _od_order
        max_V_repeated = np.repeat(max_V_per_OD, self._path_counts) # This expands it back to length totalPaths
        
        # Numerator
        exp_V = np.exp(V - max_V_repeated) # exp_V = exp(V - max_V_per_OD) 

        # Denominator
        sum_exp_per_OD = np.add.reduceat(exp_V, self._od_split_indices_0) # This gives the sum of exp_V for each OD, aligned with _od_order
        sum_exp_repeated = np.repeat(sum_exp_per_OD, self._path_counts) # This expands it back to length totalPaths
        
        # Probability = numerator / denominator
        self.P = exp_V / sum_exp_repeated
        self.Lh = self.P * self._path_demands

    def updateGap(self):
        """
        Compute the relative gap (RGAP) for the SUE and record the L2 
        flow residual for the MSA-ACS step rule.

        Each path carries a perceived generalized cost
            g_p = C_p + (1 / theta) * (1 + log(h_p)),
        which equals the gradient of the SUE objective with respect to h_p.
            RGAP = sum_p h_p * (g_p - g_min(OD(p))) / sum_p h_p * |g_p|,
        where g_min(OD(p)) is the minimum g over paths in OD(p) restricted
        to paths with positive flow. A floor of 1e-300 is applied inside the
        logarithm to guard against log(0), and zero-flow paths are excluded
        from the per-OD min by setting their g to +inf before the reduction.

        (There shouldnt be any zero flow path anyway, but this is just a safeguard 
        against numerical issues.)

        Separately, the L2 norm ||Lh - h||_2 is appended to self.prev_gap.
        This residual, not RGAP, is the convergence measure consumed by the
        adaptive step-size rule (MSA-ACS).

        Updates
        -------
        self.gap : float
            Current relative gap (RGAP).
        self.prev_gap : list of float
            History of ||Lh - h||_2 values, extended by the current iteration.
        """
                
        h_safe = np.maximum(self.h, 1e-300)
        g = self.C + (1.0 / self.theta) * (1.0 + np.log(h_safe))

        g_for_min = np.where(self.h > 1e-300, g, np.inf)
        g_min_per_OD = np.minimum.reduceat(g_for_min, self._od_split_indices_0)
        g_min_repeated = np.repeat(g_min_per_OD, self._path_counts)

        numerator = np.sum(self.h * (g - g_min_repeated))

        denominator = np.sum(self.h * np.abs(g))

        self.gap = numerator / denominator

        lh_h = np.linalg.norm(self.Lh - self.h) 
        self.prev_gap.append(lh_h) # Use lh-h as gap for MSA-ACS step rule

    def iteration0(self, networkName=None):
        """Initializes the network with a strictly positive Free-Flow Logit distribution."""
        print("Initializing network with strict Free-Flow Logit distribution...")
        
        self.h = np.zeros(self.totalPaths, dtype=FLOAT)
        self.updatePathCosts(self.h)
        self.updateLogitAndTargetFlows()
        self.h = self.Lh.copy()

        print(f"Minimum path flow after initialization: {self.h.min():.4e}")
    
    def assignment(self, accuracy, maxIter, initType, networkName, stepRule, resume=False, \
                   I_s = 10, demand_multiplier=1.0, time_limit=60*60, save_log=True, \
                    save_final_solution=False, calcBounds=False):
        """
        Execute the main path-based stochastic user equilibrium assignment loop.

        The supported step rules are:
          - MSA-HS: harmonic-step MSA, alpha = 1/it.
          - MSA-ACS: adaptive constant-step MSA driven by the recent history of
            ||Lh - h|| residuals over a window of length I_s.
          - BB1, BB2: Barzilai-Borwein steps from the current and previous
            iterates of h and Lh - h.
          - BB1-ACS, BB2-ACS: BB combined with the adaptive constant-step
            safeguard governed by I_s.

        Parameters
        ----------
        accuracy : float
            Target relative gap for convergence.
        maxIter : int
            Upper bound on the iteration counter.
        initType : str
            Initialization strategy passed to iteration0 when resume is False.
        networkName : str
            Identifier used in log and solution file paths.
        stepRule : str
            Step-size rule. One of MSA-HS, MSA-ACS, BB1, BB2, BB1-ACS, BB2-ACS.
        resume : bool, optional
            If True, continue from the current self.it without reinitializing
            or clearing self.prev_gap. Defaults to False.
        I_s : int, optional
            Window length for the adaptive constant-step safeguard used by
            MSA-ACS and the BB-ACS variants. Defaults to 10.
        demand_multiplier : float, optional
            Scalar tag included in log and solution file names to identify
            scaled-demand experiments. Defaults to 1.0.
        time_limit : float, optional
            Wall-clock cap in seconds. Defaults to 900.
        save_log : bool, optional
            If True, write the iteration log to logResults/{networkName}/.
            Defaults to True.
        save_final_solution : bool, optional
            If True and the run converges, pickle h, a, T(a), phi(h) and zeta(a)
            of the final solution to optimalSolutions/. This is the reference
            hat h for a later bound run. Defaults to False.
        calcBounds : bool, optional
            If True, evaluate the bounds after every iteration and write them to
            boundResults/. The reference hat h is read from the file that
            save_final_solution=True writes for the same network, theta,
            demand multiplier, k and stepRule. Defaults to False.

        Typical use for the bounds:
            1. assignment(accuracy=1e-12, ..., save_final_solution=True)   # reference hat h
            2. assignment(accuracy=1e-8,  ..., calcBounds=True)            # bounds

        Side Effects
        ------------
        Writes a per-iteration log file to one of:
          - logResults/{networkName}/{networkName}_theta{theta}_d{demand_multiplier}x_K{k}_{stepRule}.txt
            when save_log is True, or
        On convergence, if save_final_solution is True, pickles a dict with
        h, a, T(a), phi(h) and zeta(a) of the final solution to
        optimalSolutions/{networkName}_theta{theta}_d{demand_multiplier}x_K{k}_{stepRule}_solution.pkl.

        If calcBounds is True, writes plots.csv to
        boundResults/{networkName}/theta{theta}_d{demand_multiplier}x_K{k}_{stepRule}/.

        Prints per-iteration progress (gap, iteration, elapsed time) and a
        termination summary (total time, iteration count, final gap, final TSTT) to stdout.
        """

        
        self.networkName = networkName
        if save_log:
            log_file = f'logResults/{self.networkName}/{self.networkName}_theta{self.theta}_d{demand_multiplier}x_K{self.k}_{stepRule}.txt'
        else:
            log_file = os.devnull

        solution_file = f'optimalSolutions/{self.networkName}_theta{self.theta}_d{demand_multiplier}x_K{self.k}_{stepRule}_solution.pkl'

        with open(log_file, 'w') as f:
            f.write(f"Network: {networkName}, theta: {self.theta}, k: {self.k}\n")

        if calcBounds:
            if resume or save_final_solution:
                raise ValueError("calcBounds=True needs resume=False and save_final_solution=False")

            # Reference equilibrium, saved by an earlier run with save_final_solution=True
            with open(solution_file, 'rb') as f:
                self.boundsReference = pickle.load(f)

            bound_folder = f'boundResults/{self.networkName}/theta{self.theta}_d{demand_multiplier}x_K{self.k}_{stepRule}'
            os.makedirs(bound_folder, exist_ok=True)
            self.boundsStream = open(os.path.join(bound_folder, "plots.csv"), "w", newline="")
            self.boundsWriter = None
            self.boundsClock = None


        if not resume:
            self.iteration0(initType)
            start_it = 1
            self.prev_gap.clear()
            self.h_n_minus_one = None
            self.h_n_minus_two = None
            self.lh_n_minus_one = None
            self.lh_n_minus_two = None
        else:
            start_it = self.it + 1
            print(f"Resuming assignment from iteration {start_it}...")

        converged = False
        startP = time.time()
        last_time = time.time()

        self.updatePathCosts(self.h)
        self.updateLogitAndTargetFlows()
        self.updateGap()


        if calcBounds:
            self.recordBounds(iteration=start_it - 1)


        for it in range(start_it, maxIter):

            if self.trackHistory:
                self.history["Iteration"].append(it)
                self.history['h'].append(self.h.copy())
                self.history['Lh'].append(self.Lh.copy())
                self.history['gap'].append(self.gap)
                self.history['alpha'].append(self.alpha)

            self.it = it
            
            if stepRule in ("BB1", "BB2", "BB1-ACS", "BB2-ACS"):
                if self.h_n_minus_one is None:
                    self.alpha = 0.5
                else:
                    self.alpha = self.compute_bb_step_size(bb_type=stepRule, I_s=I_s, log_file=log_file)
            elif stepRule == "MSA-HS":
                self.alpha = 1.0 / it
            elif stepRule == "MSA-ACS":
                self.alpha = self.constantMSAStep(it, self.prev_gap, self.alpha, I_s)

            # Apply update
            self.h_n_minus_two = self.h_n_minus_one.copy() if self.h_n_minus_one is not None else None
            self.lh_n_minus_two = self.lh_n_minus_one.copy() if self.lh_n_minus_one is not None else None
            self.h_n_minus_one = self.h.copy()
            self.lh_n_minus_one = self.Lh.copy()
            self.h = (1 - self.alpha) * self.h + self.alpha * self.Lh
            self.updatePathCosts(self.h)
            self.updateLogitAndTargetFlows()
            self.updateGap()

            if calcBounds:
                self.recordBounds(iteration=it)

            time_elapsed = time.time() - last_time
            last_time = time.time()

            print(f"Current gap: {self.gap}, Iteration: {it}, Time elapsed: {time_elapsed:.2f}s")

            with open(log_file, 'a') as f:
                f.write(f"Iteration: {it}, Gap: {self.gap}, TSTT: {self.tstt}, Alpha: {self.alpha}, Time elapsed: {time_elapsed}\n")
            
            if time.time() - startP > time_limit:  
                print(f"\nWarning: Assignment has been running for over {time_limit / 60:.1f} minutes.")
                print(f"Assignment took {time.time() - startP:.2f} seconds")
                print(f"Converged in {it} iterations")
                print(f"Final gap: {self.gap:.4f}")
                print(f"Final TSTT: {self.tstt:.4f}")
                converged = False

                self.optimalSol = self.h.copy()

                with open(log_file, 'a') as f:
                    f.write("Link flows:\n")
                    for link in self.linkSet:
                        f.write(f"  Link {link}: Flow = {self.linkFlows[link]:.4f}, Cost = {self.linkCosts[link]:.4f}\n")
                break

            if self.gap < accuracy:
                print("\nDONE! Desired accuracy achieved.")
                print(f"Assignment took {time.time() - startP:.2f} seconds")
                print(f"Converged in {it} iterations")
                print(f"Final gap: {self.gap:.4f}")
                print(f"Final TSTT: {self.tstt:.4f}")
                converged = True

                self.optimalSol = self.h.copy()
                if save_final_solution:
                    # Save hat h together with everything a bound run compares against,
                    # so nothing has to be recomputed when it is loaded
                    optimalSol = {
                        "h": self.h.copy(),
                        "a": np.array([self.linkFlows[link] for link in self.linkIndexMap], dtype=FLOAT),
                        "T": np.array([self.linkCosts[link] for link in self.linkIndexMap], dtype=FLOAT),
                        "phi": self.phi(),
                        "zeta": self.tstt,
                    }
                    os.makedirs('optimalSolutions', exist_ok=True)
                    with open(solution_file, 'wb') as f:
                        pickle.dump(optimalSol, f)


                with open(log_file, 'a') as f:
                    f.write("Link flows:\n")
                    for link in self.linkSet:
                        f.write(f"  Link {link}: Flow = {self.linkFlows[link]:.4f}, Cost = {self.linkCosts[link]:.4f}\n")

                break

        if calcBounds:
            self.boundsStream.close()

        if not converged:
            print("The assignment did not converge within the maximum iterations.")
            print(f"Final gap: {self.gap}")

    def constantMSAStep(self, k, prev_gap, alpha, I_s, epsilon=1e-2):
        """Adaptive constant-step MSA rule. For the first I_s iterations, alpha = 1/k. 
        After that, if the relative decrease in the gap over the most recent 3 iterations is 
        less than epsilon, alpha is reset to 1/k; otherwise it remains unchanged."""
        if k <= I_s:
            alpha = 1.0 / k
        else:
            g0 = prev_gap[0]  # oldest of the 3
            g2 = prev_gap[-1]  # newest
            if (g0 - g2) / g0 < epsilon: 
                alpha = 1.0 / k
            # otherwise alpha stays unchanged (held constant)
        return alpha

    def compute_bb_step_size(self, bb_type, I_s, log_file=None):
            """Compute the Barzilai-Borwein step size based on the current and previous iterates."""
            s = self.h - self.h_n_minus_one
            t = self.Lh - self.lh_n_minus_one
            
            y = s - t

            if bb_type == "BB1":
                alpha = (s.T @ y) / (y.T @ y)
                alpha = np.clip(alpha, 0.0, 1.0)
                # The paper says BB1 can never have nan, so we code eaxtly this
                # Even if there is nan, we will report that in the log and then just return the nan.
                return alpha

            elif bb_type == "BB2":
                alpha = ((s.T @ s)) / (s.T @ y)
                alpha = np.clip(alpha, 0.0, 1.0)
                # The paper says BB2 can never have nan, so we code eaxtly this
                # Even if there is nan, we will report that in the log and then just return the nan.
                return alpha
            
            elif bb_type == "BB1-ACS":
                alpha = (s.T @ y) / (y.T @ y)
                alpha = np.clip(alpha, 0.0, 1.0) 
                if np.isnan(alpha):
                    alpha = self.constantMSAStep(self.it, self.prev_gap, self.alpha, I_s)
                return alpha
            
            elif bb_type == "BB2-ACS":
                alpha = ((s.T @ s)) / (s.T @ y)
                alpha = np.clip(alpha, 0.0, 1.0)
                if np.isnan(alpha):
                    alpha = self.constantMSAStep(self.it, self.prev_gap, self.alpha, I_s)
                return alpha

            else:
                raise TypeError(f"Unknown BB type: {bb_type}")
                
    ####################################################
    # Bound section
    ####################################################

    def computeWl(self):
            """
            W_l = OD pairs w with at least one path using link l and at least one path not using it.
            Returns a list where W_l[l] is the list of OD pair indices w in W_l
            (l is the row index of the link in D, w is the position of the OD pair in self._od_order).
            """
            W_l = [[] for link in self.linkIndexMap]
    
            for w, OD in enumerate(self._od_order):
                paths = self.allPathsODCache[OD]
    
                # numPathsUsing[link] = how many paths of w use this link
                numPathsUsing = {}
                for path in paths:
                    for j in range(len(path) - 1):
                        link = (path[j], path[j + 1])
                        if link in numPathsUsing:
                            numPathsUsing[link] = numPathsUsing[link] + 1
                        else:
                            numPathsUsing[link] = 1
    
                # w is in W_l when at least one path uses l (count >= 1)
                # and at least one path does not use l (count < number of paths)
                for link, count in numPathsUsing.items():
                    if count >= 1 and count < len(paths):
                        W_l[self.linkIndexMap[link]].append(w)
    
            return W_l
    
    def computeSl(self):
        """S_l = sum of d_w over w in W_l, one value per link (same order as the rows of D)."""
        S_l = np.zeros(len(self.W_l), dtype=FLOAT)
        for l in range(len(self.W_l)):
            for w in self.W_l[l]:
                S_l[l] = S_l[l] + self.d_w[w]
        return S_l

    # ---------- Objective functions ----------

    def beta(self):
        """
        beta(h) = sum_l integral_0^{a_l} tau_l(x) dx at the current link flows.
        For BPR: integral_0^a tau_l = (tau0 + toll) a + tau0 alpha a^(p+1) / ((p+1) c^p).
        """
        total = 0.0
        for link, l in self.linkSet.items():
            a_l = self.linkFlows[link]
            total = total + (l.fft + l.tollInTime) * a_l \
                            + l.fft * l.alpha * a_l ** (l.beta + 1) / ((l.beta + 1) * l.capacity ** l.beta)
        return total

    def eta(self, g):
        """eta(g) = (1/theta) sum_i g_i ln g_i."""
        return np.sum(g * np.log(g)) / self.theta

    def phi(self):
        """SUE objective phi(h) = beta(h) + eta(h) at the current h."""
        return self.beta() + self.eta(self.h)

    def psi(self, g):
        """psi_h(g) = beta(h) + C(h)^T (g - h) + eta(g), with h the current flows."""
        return self.beta() + np.dot(self.C, g - self.h) + self.eta(g)

    def gamma(self):
        """gamma(h) = psi_h(h) - psi_h(L(h))   (Theorem 3_objectivegap)."""
        return self.psi(self.h) - self.psi(self.Lh)

  
    # ---------- All bounds at the current h ----------

    def evaluateBounds(self):
        """
        Evaluate every bound of Section 3 at the solver's current h.

        values["times"] holds the wall-clock seconds of each bound:
            gamma          Theorem 3_objectivegap
            m_h            Theorem 3_globalpathbound
                       link_flow      Lemma 3_linkbound and Theorem 3_linkbound
            link_cost      Corollary 3_linkcostbound
            tstt           Theorem 3_tsttbound and Corollary 3_tsttconvex
            link_cost_lemma, tstt_lemma   the same bounds with tilde m_a^l of Lemma 3_linkbound
                                          in place of m_a^l (before the strengthening)
        phi, RGAP and ||L(h) - h|| are diagnostics and are not timed.
        """
        times = {}

        # Solver values as arrays, in the same order as the rows of D
        a = np.array([self.linkFlows[link] for link in self.linkIndexMap], dtype=FLOAT)
        T = np.array([self.linkCosts[link] for link in self.linkIndexMap], dtype=FLOAT)
        T_prime = np.array([self.linkCostDerivatives[link] for link in self.linkIndexMap], dtype=FLOAT)

        # Theorem : objective gap
        start = time.perf_counter()
        gamma = self.gamma()
        times["gamma"] = time.perf_counter() - start

        # Theorem : path bound : m_h = sqrt(theta d_max gamma)
        start = time.perf_counter()
        m_h = np.sqrt(self.theta * self.d_max * gamma)
        times["m_h"] = time.perf_counter() - start

        # Lemma : link flow bound : tilde m_a^l = sqrt(theta S_l gamma / 2)
        # Theorem : link flow bound : R_a^l = min of tau_l' over {x >= 0 : |x - a_l| <= tilde m_a^l}.
        # tau_l is convex for BPR, so tau_l' is nondecreasing and the min is at the left end max(0, a_l - tilde m_a^l).
        # m_a^l = tilde m_a^l / sqrt(1 + theta S_l R_a^l / 4)
        start = time.perf_counter()
        m_a_tilde = np.sqrt(self.theta * self.S_l * gamma / 2)
        R_a = []
        for i, link in enumerate(self.linkIndexMap):
            R_a.append(self._linkCostDerivative(link, max(0, a[i] - m_a_tilde[i])))
        R_a = np.array(R_a, dtype=FLOAT)
        m_a = m_a_tilde / np.sqrt(1 + self.theta * self.S_l * R_a / 4)
        times["link_flow"] = time.perf_counter() - start


        # Corollary : link cost bound : L_a^l = max of tau_l' over A_a^l = [max(0, a_l - m_a^l), a_l + m_a^l].
        # tau_l' is nondecreasing for BPR, so the max is at the right end a_l + m_a^l.
        start = time.perf_counter()
        L_a = []
        for i, link in enumerate(self.linkIndexMap):
            L_a.append(self._linkCostDerivative(link, a[i] + m_a[i]))
        L_a = np.array(L_a, dtype=FLOAT)
        linkCostBound = L_a * m_a
        times["link_cost"] = time.perf_counter() - start

        # Theorem : TSTT Bound and Corollary : Convex TSTT Bound
        start = time.perf_counter()
        zeta = self.tstt

        # q = D^T [T(a) + T'(a) a]
        q = self.fullLinkPathIncidence_T @ (T + T_prime * a)

        # Q_a = sqrt( sum_w d_w (max_{i in P_w} q_i - min_{i in P_w} q_i)^2 )
        qMax = np.maximum.reduceat(q, self._od_split_indices_0)
        qMin = np.minimum.reduceat(q, self._od_split_indices_0)
        Q_a = np.sqrt(np.sum(self.d_w * (qMax - qMin) ** 2))

        # Corollary 3_tsttconvex on its own: it needs only q, Q_a and gamma
        linearTerm = Q_a * np.sqrt(self.theta * gamma / 2)
        tsttLowerConvex = zeta - linearTerm
        times["tstt_convex"] = time.perf_counter() - start

        # K_a^l = max over A_a^l of |2 tau_l'(x) + x tau_l''(x)|.
        # For BPR, x tau_l''(x) = (p_l - 1) tau_l'(x), so 2 tau_l' + x tau_l'' = (p_l + 1) tau_l',
        # which is nonnegative and nondecreasing: the max is at the right end, (p_l + 1) L_a^l.
        K_a = (self.p_l + 1) * L_a

        tsttBound = linearTerm + np.sum(K_a * m_a ** 2) / 2
        times["tstt"] = time.perf_counter() - start

        # The link cost and TSTT bounds with tilde m_a^l of Lemma 3_linkbound in place of m_a^l,
        # i.e. before the strengthening in Theorem 3_linkbound (fig:4_bounds_cs).
        # L_a^l and K_a^l are then taken over the interval [max(0, a_l - tilde m_a^l), a_l + tilde m_a^l].
        start = time.perf_counter()
        L_a_tilde = []
        for i, link in enumerate(self.linkIndexMap):
            L_a_tilde.append(self._linkCostDerivative(link, a[i] + m_a_tilde[i]))
        L_a_tilde = np.array(L_a_tilde, dtype=FLOAT)
        linkCostBoundTilde = L_a_tilde * m_a_tilde
        times["link_cost_lemma"] = time.perf_counter() - start

        start = time.perf_counter()
        K_a_tilde = (self.p_l + 1) * L_a_tilde
        tsttBoundTilde = linearTerm + np.sum(K_a_tilde * m_a_tilde ** 2) / 2
        times["tstt_lemma"] = time.perf_counter() - start

        # Diagnostics (not timed)
        phi = self.phi()
        residual = np.linalg.norm(self.Lh - self.h)     # ||L(h) - h||

        rgap = self.gap

        return {
            "h": self.h.copy(), "a": a, "T": T,
            "phi": phi, "gamma": gamma, "rgap": rgap, "residual": residual,
            "m_h": m_h,
            "m_a_tilde": m_a_tilde, "R_a": R_a,
            "m_a": m_a, "L_a": L_a, "linkCostBound": linkCostBound,
            "Q_a": Q_a, "K_a": K_a, "zeta": zeta,
            "tsttBound": tsttBound, "tsttLowerConvex": tsttLowerConvex,
            "linkCostBoundTilde": linkCostBoundTilde, "tsttBoundTilde": tsttBoundTilde,
            "times": times,
        }

    # ---------- Bound run (called from assignment when calcBounds=True) ----------

    def recordBounds(self, iteration):
        """
        Called by assignment after every iteration when calcBounds=True.
        Saves everything for this iteration as one row of plots.csv:
        the bounds at the current h, the actual errors against the reference hat h, and timings.
            bound_time: time of evaluateBounds
            iter_time:  solver time since the previous call (one iteration without the bounds)
        """

        if self.boundsClock is None:
            iterationTime = 0.0
        else:
            iterationTime = time.perf_counter() - self.boundsClock

        # Bounds at the current h
        start = time.perf_counter()
        values = self.evaluateBounds()
        boundTime = time.perf_counter() - start

        # Actual errors against the reference hat h, with v = h - hat h
        ref = self.boundsReference
        
        v = values["h"] - ref["h"]

        v_l1_per_OD = np.add.reduceat(np.abs(v), self._od_split_indices_0)    # ||h^w - hat h^w||_1

        row = {
            "iteration": iteration,
            "rgap": values["rgap"],

            # Objective sandwich: G <= phi(h) - phi(hat h) <= gamma(h)
            "objective_growth": np.sum(v_l1_per_OD ** 2 / self.d_w) / (2 * self.theta),    # G
            # "delta_phi": values["phi"] - ref["phi"],
            "delta_phi": values["phi"] - ref["phi"],
            "gamma": values["gamma"],

            # Actual errors
            "error_h_l2": np.linalg.norm(v),
            "error_a_l2": np.linalg.norm(values["a"] - ref["a"]),
            "error_T_l2": np.linalg.norm(values["T"] - ref["T"]),
            "error_zeta": np.abs(ref["zeta"] - values["zeta"]),
            "max_error_a": np.max(np.abs(values["a"] - ref["a"])),
            "max_error_T": np.max(np.abs(values["T"] - ref["T"])),

            # Bounds
            "m_h": values["m_h"],
            "M_L": np.linalg.norm(values["m_a"]),
            "M_C": np.linalg.norm(values["linkCostBound"]),
            "m_zeta": values["tsttBound"],
            "zeta_lower_convex": values["tsttLowerConvex"],
            "max_m_l": np.max(values["m_a"]),
            "max_Lm": np.max(values["linkCostBound"]),

            # The same bounds with tilde m_a^l of Lemma 3_linkbound (before the strengthening)
            "M_L_lemma": np.linalg.norm(values["m_a_tilde"]),
            "M_C_lemma": np.linalg.norm(values["linkCostBoundTilde"]),
            "m_zeta_lemma": values["tsttBoundTilde"],
            "max_m_l_lemma": np.max(values["m_a_tilde"]),
            "max_Lm_lemma": np.max(values["linkCostBoundTilde"]),

            # Values at h and at hat h (for relative errors)
            "phi": values["phi"],
            "zeta": values["zeta"],
            "residual_l2": values["residual"],
            "hat_phi": ref["phi"],
            "hat_zeta": ref["zeta"],
            "hat_h_l2": np.linalg.norm(ref["h"]),
            "hat_a_l2": np.linalg.norm(ref["a"]),
            "hat_T_l2": np.linalg.norm(ref["T"]),

            # Timing
            "bound_time": boundTime,
            "iter_time": iterationTime,
        }
        for name, seconds in values["times"].items():
            row["time_" + name] = seconds

        if self.boundsWriter is None:
            self.boundsWriter = csv.DictWriter(self.boundsStream, fieldnames=list(row))
            self.boundsWriter.writeheader()
        self.boundsWriter.writerow(row)
        self.boundsStream.flush()

        self.boundsClock = time.perf_counter()
