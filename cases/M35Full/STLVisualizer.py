import pyvista as pv
import numpy as np


triangleList = [27088, 83829, 84185, 83589, 83587, 622]


filename = "M-Jet_35_pump_main.STL"

def show_stl(file_path, triangle_indices):
    # 1) Read the STL file
    mesh = pv.read(file_path)
    
    # 2) Print coordinates of vertices for the specified triangles
    # In PyVista, 'cells' represent the triangles. 
    # mesh.faces is a 1D array: [3, v1, v2, v3, 3, v4, v5, v6...] 
    # because it's a triangular mesh.
    print(f"--- Coordinates for indices: {triangle_indices} ---")
    
    for idx in triangle_indices:
        # Get the point indices for this specific triangle (cell)
        cell_point_ids = mesh.get_cell(idx).point_ids
        coords = mesh.points[cell_point_ids]
        
        print(f"Triangle {idx} vertices:")
        for i, pt in enumerate(coords):
            print(f"  Vertex {i}: {pt}")
        print("-" * 20)

    # 3) Create a sub-mesh containing ONLY the listed triangles
    # 'extract_cells' is the magic tool you were looking for
    subset_mesh = mesh.extract_cells(triangle_indices)

    # 4) Display the meshes
    # We use a plotter to show the original (ghosted) and the selection (solid)
    plotter = pv.Plotter()
    
    # Show the full mesh in wireframe/opacity so we can see context
    plotter.add_mesh(mesh, color="white", opacity=0.1, style='wireframe')
    
    # Show only the triangles in your list in a bold color
    plotter.add_mesh(subset_mesh, color="red", label="Selected Triangles")
    
    plotter.add_legend()
    plotter.show()

show_stl(filename, triangleList)
