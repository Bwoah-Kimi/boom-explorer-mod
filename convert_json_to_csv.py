#!/usr/bin/env python3
"""
Convert DSE JSON results to CSV format for BOOM Explorer

Usage:
    python convert_json_to_csv.py <input.json> <output.csv>
"""

import json
import csv
import sys


def convert_json_to_csv(json_file, csv_file):
    """Convert JSON DSE results to CSV format"""
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get design space parameters (keys to encode)
    design_space = data['design_space']
    param_names = list(design_space.keys())
    
    # Create encoding mappings for each parameter
    encodings = {}
    for param, values in design_space.items():
        encodings[param] = {val: idx for idx, val in enumerate(values)}
    
    # Prepare CSV data
    rows = []
    for result in data['iteration_results']:
        dse_config = result['dse_config']
        objectives = result['objectives']
        
        # Encode design parameters
        encoded_params = []
        for param in param_names:
            value = dse_config[param]
            encoded_params.append(encodings[param][value])
        
        # Create row: space-separated params, then objectives
        params_str = ' '.join(map(str, encoded_params))
        row = [params_str] + objectives
        rows.append(row)
    
    # Write to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: params, objective names
        # Assuming 2 objectives: performance (cycles) and cost
        header = ['params', 'total_cycles', 'total_cost']
        writer.writerow(header)
        
        # Write all rows
        writer.writerows(rows)
    
    print(f"Converted {len(rows)} designs from {json_file} to {csv_file}")
    print(f"Design space parameters: {param_names}")
    print(f"Encodings:")
    for param, mapping in encodings.items():
        print(f"  {param}: {mapping}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_json_to_csv.py <input.json> <output.csv>")
        print("Example: python convert_json_to_csv.py large_design_space_enum.json my_dataset.csv")
        sys.exit(1)
    
    json_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    convert_json_to_csv(json_file, csv_file)
