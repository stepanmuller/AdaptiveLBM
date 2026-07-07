import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from collections import Counter

def simulate_partitioning(i_span, j_span, limit_per_thread):
    total_tasks = i_span * j_span
    num_threads = (total_tasks + limit_per_thread - 1) // limit_per_thread
    
    # List to store (i, j, thread_id) for verification and plotting
    computed_cells = []
    
    # Simulate the GPU threads
    for thread_index in range(num_threads):
        task_start = thread_index * limit_per_thread
        task_end = min(task_start + limit_per_thread, total_tasks) - 1
        
        if task_start >= total_tasks:
            continue
            
        # --- YOUR EXACT MATH (UNTOUCHED) ---
        j_start_thread = task_start // i_span
        i_start_thread = task_start % i_span
        
        j_end_thread = (task_end // i_span) + 1
        i_end_thread = (task_end % i_span) + 1
        
        for j in range(j_start_thread, j_end_thread):
            if j == j_start_thread:
                i_start_j = i_start_thread
            else:
                i_start_j = 0
                
            if j == j_end_thread - 1:
                i_end_j = i_end_thread
            else:
                i_end_j = i_span
                
            for i in range(i_start_j, i_end_j):
                computed_cells.append((i, j, thread_index))
                
    return computed_cells

def plot_grid(i_span, j_span, computed_cells, missing, duplicates, oob):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot successfully computed cells
    x = [cell[0] for cell in computed_cells]
    y = [cell[1] for cell in computed_cells]
    thread_ids = [cell[2] for cell in computed_cells]
    
    ax.scatter(x, y, c=thread_ids, cmap='tab20', marker='s', s=1500, edgecolors='black', alpha=0.8)
    
    # Label each computed cell
    for (i, j, t_id) in computed_cells:
        color = 'red' if (i, j) in duplicates else 'white'
        prefix = "DUP\nT:" if (i, j) in duplicates else "T:"
        ax.text(i, j, f"{prefix}{t_id}", ha='center', va='center', color=color, 
                fontweight='bold', fontsize=9,
                path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    # Plot missing cells (Red X)
    for (i, j) in missing:
        ax.scatter(i, j, color='white', marker='s', s=1500, edgecolors='red', linewidth=3, hatch='//')
        ax.text(i, j, "MISSING", ha='center', va='center', color='red', fontweight='bold', fontsize=10)

    # Plot out of bounds cells
    for (i, j) in oob:
        ax.scatter(i, j, color='black', marker='s', s=1500, edgecolors='red', linewidth=2)
        ax.text(i, j, "OOB", ha='center', va='center', color='red', fontweight='bold')

    # Calculate plot limits to include Out-of-Bounds cells if they exist
    min_x, max_x = -0.5, i_span - 0.5
    min_y, max_y = -0.5, j_span - 0.5
    if oob:
        min_x = min(min_x, min(c[0] for c in oob) - 0.5)
        max_x = max(max_x, max(c[0] for c in oob) + 0.5)
        min_y = min(min_y, min(c[1] for c in oob) - 0.5)
        max_y = max(max_y, max(c[1] for c in oob) + 0.5)

    # Grid formatting
    ax.set_xticks(np.arange(min_x, max_x + 1, 1), minor=True)
    ax.set_yticks(np.arange(min_y, max_y + 1, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', length=0)
    
    plt.title(f"Grid Partitioning (Width={i_span}, Height={j_span})\nLimit={TASKS_PER_THREAD}/thread", fontsize=14, pad=15)
    plt.xlabel("i (Columns)", fontsize=12)
    plt.ylabel("j (Rows)", fontsize=12)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# --- Run the Simulation ---
I_SPAN = 10
J_SPAN = 6
TASKS_PER_THREAD = 13

print(f"--- SIMULATION START ---")
print(f"Target Area: {I_SPAN}x{J_SPAN} ({I_SPAN * J_SPAN} cells)")

cells = simulate_partitioning(I_SPAN, J_SPAN, TASKS_PER_THREAD)

# --- ERROR ANALYSIS ---
expected_cells = set((i, j) for i in range(I_SPAN) for j in range(J_SPAN))
computed_coords = [(cell[0], cell[1]) for cell in cells]
computed_set = set(computed_coords)

missing = sorted(list(expected_cells - computed_set))
oob = sorted(list(computed_set - expected_cells))

counts = Counter(computed_coords)
duplicates = {k: v for k, v in counts.items() if v > 1}

print("\n--- DIAGNOSTICS ---")
if not missing and not oob and not duplicates:
    print("SUCCESS: Perfect partitioning! All cells hit exactly once.")
else:
    if missing:
        print(f"[!] MISSING CELLS ({len(missing)}): These were skipped entirely.")
        print(f"    -> {missing}")
    if oob:
        print(f"[!] OUT OF BOUNDS ({len(oob)}): Thread processed outside the target grid.")
        print(f"    -> {oob}")
    if duplicates:
        print(f"[!] DUPLICATE CELLS ({len(duplicates)}): These were computed multiple times.")
        print(f"    -> {list(duplicates.keys())}")

plot_grid(I_SPAN, J_SPAN, cells, missing, duplicates, oob)
