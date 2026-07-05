# LBM D3Q27 Velocity Set
cx = [ 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 0, 0,-1, 1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1 ]
cy = [ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1 ]
cz = [ 0, 0, 0,-1, 1, 0, 0,-1, 1, 1,-1, 0, 0,-1, 1, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1,-1, 1 ]

# Helper function to find indices based on a target value (1 for pos, -1 for neg)
def get_indices(axis_list, target_value):
    return [i for i, val in enumerate(axis_list) if val == target_value]

# Map out our directions and their corresponding lists
distributions = {
    "positiveCx": get_indices(cx, 1),
    "negativeCx": get_indices(cx, -1),
    "positiveCy": get_indices(cy, 1),
    "negativeCy": get_indices(cy, -1),
    "positiveCz": get_indices(cz, 1),
    "negativeCz": get_indices(cz, -1),
}

# Print them out in C/C++ array format
for name, indices in distributions.items():
    indices_str = ", ".join(map(str, indices))
    # Using standard C/C++ array declaration syntax
    print(f"const int {name}Distributions[{len(indices)}] = {{ {indices_str} }};")
