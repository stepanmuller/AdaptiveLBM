import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma

# Read the rotating frame velocity constant from command line arguments
# Defaults to 0.0 if not provided
try:
    v_frame = float(sys.argv[1])
except IndexError:
    v_frame = 0.0

# 1. Read binary data
file_path = "/dev/shm/sim_data.bin"
with open(file_path, "rb") as f:
    plotNumber = np.fromfile(f, dtype=np.int32, count=1)[0]
    # Header: [plotNumber, nVertical, nHorizontal, n_vars]
    dims = np.fromfile(f, dtype=np.int32, count=3)
    nVertical, nHorizontal, n_vars = dims
    data = np.fromfile(f, dtype=np.float32).reshape((nVertical, nHorizontal, n_vars))

# 2. Extract variables (Matches C++ fwrite order)
p = data[:, :, 0]
uHorizontal = data[:, :, 1]
uVertical = data[:, :, 2]
uNormal = data[:, :, 3]
mask = data[:, :, 4] 
gridID = data[:, :, 5]

# Absolute planar velocity
uPlanar = np.sqrt(uHorizontal**2 + uVertical**2)

# Relative planar velocity (rotating frame)
uVertical_rel = uVertical + v_frame
uPlanar_rel = np.sqrt(uHorizontal**2 + uVertical_rel**2)

# 3. Setup Figure (FIXED: 3 plots side-by-side)
fig = plt.figure(figsize=(20, 5), constrained_layout=True)
gs = fig.add_gridspec(1, 5)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax3 = fig.add_subplot(gs[3])
ax4 = fig.add_subplot(gs[4])

is_solid = mask > 0.5 

def setup_plot(ax, data_array, label):
	# Get only the fluid data (ignore solid mask for scale calculation)
    fluid_data = data_array[mask <= 0.5]
    
    # Calculate 5th and 95th percentiles (removes 2% smallest and 2% largest)
    vmin = np.nanpercentile(fluid_data, 1)
    vmax = np.nanpercentile(fluid_data, 99)
    
    # If the range is zero (constant field), default to data min/max
    if (vmin == vmax) or label == "Grid ID [1]":
        vmin, vmax = np.min(fluid_data), np.max(fluid_data)
        
    masked_data = ma.array(data_array, mask=is_solid)
    img = ax.imshow(masked_data, origin="lower", cmap="viridis", aspect="equal", interpolation="nearest")
    # REMOVE THE FRAME (SPINES)
    for spine in ax.spines.values():
        spine.set_visible(False)
    img.cmap.set_bad(color="black")
    if vmin is not None: img.set_clim(vmin, vmax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.set_label(label)
    return img

# Plot 1: Planar Velocity + Streamlines
setup_plot(ax0, uPlanar, "Planar velocity absolute [m/s]")
horizontalVals = np.arange(nHorizontal)
verticalVals = np.arange(nVertical)
uH_stream = np.where(is_solid, np.nan, uHorizontal)
uV_stream = np.where(is_solid, np.nan, uVertical)
ax0.streamplot(horizontalVals, verticalVals, uH_stream, uV_stream, color="white", linewidth=0.3, density=1.2, arrowsize=0.4)
# Enforce strict axes limits to prevent streamline arrows from modifying the bounding box
ax0.set_xlim(-0.5, nHorizontal - 0.5)
ax0.set_ylim(-0.5, nVertical - 0.5)

# Plot 2: Relative Planar Velocity + Streamlines
setup_plot(ax1, uPlanar_rel, f"Planar velocity relative [m/s]")
uV_stream_rel = np.where(is_solid, np.nan, uVertical_rel)
ax1.streamplot(horizontalVals, verticalVals, uH_stream, uV_stream_rel, color="white", linewidth=0.3, density=1.2, arrowsize=0.4)
# Enforce strict axes limits here as well
ax1.set_xlim(-0.5, nHorizontal - 0.5)
ax1.set_ylim(-0.5, nVertical - 0.5)

# Plot 3: Normal Velocity
setup_plot(ax2, uNormal, "Normal velocity [m/s]")

# Plot 4: Pressure
setup_plot(ax3, p, "Static pressure [Pa]")

# Plot 5: Grid ID
setup_plot(ax4, gridID, "Grid ID [1]")

# 6. Save
os.makedirs("results", exist_ok=True)
plt.savefig(f"results/{plotNumber}.png", dpi=min(1000, max([300, nVertical/2, nHorizontal/2])), bbox_inches="tight")
plt.close()
