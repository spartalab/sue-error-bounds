"""
check_precision.py
Shows which floating-point types this machine really has, how precise each one is,
whether scipy sparse products work with it, and how fast it is.
Run:  python check_precision.py
"""
import platform
import time
import numpy as np
import scipy
from scipy.sparse import csr_matrix

print(f"Machine: {platform.system()} {platform.machine()}, numpy {np.__version__}, scipy {scipy.__version__}\n")

# Every float type numpy may offer (float128 only exists on some machines)
candidates = ["float16", "float32", "float64", "longdouble", "float128"]

# Small test problem: sparse matrix times vector, plus exp and log (what SUEPath does most).
# D has 10 random links per path, built directly so it needs little memory.
n_paths, n_links, linksPerPath = 200_000, 2_000, 10
rng = np.random.default_rng(0)
rows = rng.integers(0, n_links, n_paths * linksPerPath)
cols = np.repeat(np.arange(n_paths), linksPerPath)
D64 = csr_matrix((np.ones(n_paths * linksPerPath), (rows, cols)), shape=(n_links, n_paths))
h64 = rng.random(n_paths) + 0.1

print(f"{'type':<11}{'bits':>5}{'digits':>8}{'eps':>12}   {'sparse @ works':<16}{'time (s)':>9}")
print("-" * 65)
for name in candidates:
    if not hasattr(np, name):
        print(f"{name:<11}  not available on this machine")
        continue
    dtype = getattr(np, name)
    info = np.finfo(dtype)

    # Does a sparse matrix-vector product keep this type?
    h = h64.astype(dtype)
    try:
        D = D64.astype(dtype)
        sparseWorks = (D @ h).dtype == dtype
    except Exception:
        sparseWorks = False

    # Rough speed: 5 rounds of D h, exp, log
    start = time.perf_counter()
    for _ in range(5):
        x = D @ h if sparseWorks else D64 @ h64.astype(np.float64)
        np.exp(-x)
        np.log(h)
    seconds = (time.perf_counter() - start) / 5

    print(f"{name:<11}{info.bits:>5}{info.precision:>8}{float(info.eps):>12.1e}   {str(sparseWorks):<16}{seconds:>9.4f}")

longEps = float(np.finfo(np.longdouble).eps)
print()
if longEps < 1e-18:
    print("np.longdouble is real 80-bit extended precision here (about 19 digits).")
    print("Use FLOAT = np.longdouble for the bound runs that need it, np.float64 otherwise.")
else:
    print("np.longdouble is just float64 on this machine: it gives no extra precision.")
    print("Use FLOAT = np.float64 here, and run the long double runs on a Linux x86 machine.")