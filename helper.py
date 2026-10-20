import random
import pandas as pd
import re

def dict_difference(dict1, dict2):
    """
    Subtracting dict2 from dict1 key-wise
    """
    return {key: dict1[key] - dict2.get(key, 0) for key in dict1}

def dict_norm(d):
    """
    Norm of a dictionary
    """
    return sum([v**2 for v in d.values()])**0.5

def distribute_demand(demand, n):
    """
    Creates a random distribution of a given demand into n parts.
    """

    random_numbers = [random.uniform(0, demand) for _ in range(n - 1)]
    cut_points = [0] + sorted(random_numbers) + [demand]
    distribution = [cut_points[i+1] - cut_points[i] for i in range(n)]
    return distribution


def convert_network_tntp_to_dat(input_file, output_file):
    """
    Converts a TNTP network file to a DAT file format.
    This function reads a TNTP network file, extracts relevant columns, and writes them to a DAT file.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Find the data start point
    start_index = next(i for i, line in enumerate(lines) if line.startswith("~")) + 1
    data_lines = lines[start_index:]
    
    # Parse data
    data = []
    for line in data_lines:
        line = line.strip().strip(';')
        if not line:
            continue
        values = line.split()
        data.append(values[:8])  # Extract relevant columns
    
    # Create DataFrame with correct column names and order
    df = pd.DataFrame(data, columns=["origin", "dest", "capacity", "length", "fft", "alpha", "beta", "speedLimit"])
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Network file converted and saved to {output_file}")

def convert_demand_tntp_to_dat(input_file, output_file):
    """
    Converts a TNTP demand file to a DAT file format.
    This function reads a TNTP demand file, extracts relevant columns, and writes them to a DAT file.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Extract demand data
    data = []
    origin = None
    for line in lines:
        line = line.strip()
        
        if line.startswith("Origin"):
            origin = int(line.split()[1])
        elif origin and ':' in line:
            pairs = re.findall(r'(\d+)\s*:\s*([\d\.]+)', line)
            for dest, demand in pairs:
                data.append([origin, int(dest), float(demand)])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["origin", "dest", "demand"])
    df.to_csv(output_file, sep='\t', index=False)
    # print mean demand and total demand
    print(f"Total demand: {df['demand'].sum()}")
    print(f"Mean demand: {df['demand'].mean()}")
    print(f"Demand file converted and saved to {output_file}")


