import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def set_smart_ylim(ax, data):
    """Sets y-limits based on the second half of data with a 5% margin."""
    if len(data) < 2:
        return
    mid = len(data) // 2
    recent_data = data[mid:]
    y_min, y_max = np.min(recent_data), np.max(recent_data)
    padding = (y_max - y_min) * 0.05
    if padding == 0: padding = 1e-7
    ax.set_ylim(y_min - padding, y_max + padding)

def add_average_diagnostics(ax, iterations, data, unit_str="", fmt=".3f"):
    """Calculates last 20% average, draws the line, and returns the formatted string."""
    count = len(data)
    window = max(1, count // 5)
    avg_val = np.mean(data[-window:])
    
    ax.hlines(y=avg_val, xmin=iterations[-window], xmax=iterations[-1], 
              colors='black', linestyles='--', linewidth=2.0, alpha=0.9, zorder=3)
    
    return f"{avg_val:{fmt}}{unit_str}"

def plot_history(file_number):
    try:
        with open("/dev/shm/historyData.bin", "rb") as f:
            count = np.fromfile(f, dtype=np.int32, count=1)[0]
            # Read the 3 arrays sequentially
            data1 = np.fromfile(f, dtype=np.float32, count=count)
            data2 = np.fromfile(f, dtype=np.float32, count=count)
            data3 = np.fromfile(f, dtype=np.float32, count=count)
    except Exception:
        return

    # 3 stacked subplots sharing the x-axis, maintaining the 16:9 overall aspect ratio
    fig, axs = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    iterations = np.arange(count)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8)

    datasets = [data1, data2, data3]
    
    # --- Define your 3 custom titles and y-labels here ---
    titles = [
        r"\textbf{Inlet Power History}",
        r"\textbf{Mass Flow History}",
        r"\textbf{Torque History}"
    ]
    ylabels = [
        r"$P_{inlet}$ [W]",
        r"$\dot m_p$ [kg/s]",
        r"$T$ [Nm]"
    ]

    for i, ax in enumerate(axs):
        data = datasets[i]
        
        # --- Iterative Plot ---
        ax.plot(iterations, data, color='black', linewidth=1.2, alpha=0.7)
        ax.set_ylabel(ylabels[i])
        
        # Only set the x-label on the bottom plot 
        if i == 2:
            ax.set_xlabel(r"Iteration")
            
        ax.set_title(titles[i])
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        
        set_smart_ylim(ax, data)
        
        label_val = add_average_diagnostics(ax, iterations, data)
        
        # Synchronize the text box label with the y-axis label
        ax.text(0.98, 0.95, rf'\textbf{{{ylabels[i]}: {label_val}}}', 
                transform=ax.transAxes, ha='right', va='top', bbox=bbox_props)

    plt.tight_layout()
    os.makedirs("results/history", exist_ok=True)
    filename = str(file_number) + "History.png"
    plt.savefig("results/history/" + filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Get the argument from the command line, defaulting to 0 if none is provided
    file_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    plot_history(file_num)
