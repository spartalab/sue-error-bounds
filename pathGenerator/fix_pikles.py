import pickle
import os
import glob


def fix_cache_file(file_path):
    print(f"Fixing {file_path}...")

    with open(file_path, 'rb') as f:
        cache = pickle.load(f)
    print(f"  Loaded cache with {len(cache)} OD pairs. Converting node IDs to strings...")

    # Build a lookup table of every unique node ID once.
    # Then all conversions are dict lookups instead of str() calls.
    unique_nodes = set()
    for (o, d), paths in cache.items():
        unique_nodes.add(o)
        unique_nodes.add(d)
        for path in paths:
            unique_nodes.update(path)
    print(f"  Found {len(unique_nodes)} unique node IDs. Building lookup table...")

    s = {n: str(n) for n in unique_nodes}

    fixed_cache = {
        (s[o], s[d]): [[s[n] for n in path] for path in paths]
        for (o, d), paths in cache.items()
    }
    print(f"  Conversion complete. Saving fixed cache...")

    with open(file_path, 'wb') as f:
        pickle.dump(fixed_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved fixed cache to {file_path}.")

    print(f"  Done. {len(fixed_cache)} OD pairs converted.")


def fix_all_caches(folder='Paths'):
    files = glob.glob(os.path.join(folder, 'allPathsODCache_*.pkl'))
    if not files:
        print(f"No cache files found in {folder}")
        return

    print(f"Found {len(files)} cache file(s).\n")
    for file_path in files:
        print(f"Processing {file_path}...")
        # if file_path == "Paths/allPathsODCache_20_Winnipeg.pkl":
        #     fix_cache_file(file_path)
        fix_cache_file(file_path)
    print("\nAll files fixed.")


if __name__ == "__main__":
    fix_all_caches('Paths')