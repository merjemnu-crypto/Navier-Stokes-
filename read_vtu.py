import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

# --- Locate the VTU file relative to the script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
#vtu_path = os.path.join(script_dir, 'flowfield.vtu')
vtu_path = os.path.join(script_dir, 'flowfield_Re60_timestep.vtu')

# --- Read the flow field ---
mesh = pv.read(vtu_path)

# --- Visualize velocity magnitude on the mesh ---
mesh.plot(scalars='u', smooth_shading=True)  


# Load your flow field
mesh = pv.read("nekatarpp/flowfield_Re360.vtu")  

# Compute derivatives of velocity field
grads = mesh.compute_derivative(scalars="u", gradient=True)  # div u
dudx = grads['gradient'][:, 0]
dudy = grads['gradient'][:, 1]

grads = mesh.compute_derivative(scalars="v", gradient=True)  # div v
dvdx = grads['gradient'][:, 0]
dvdy = grads['gradient'][:, 1]



points = mesh.points
x, y = points[:,0], points[:,1]

u_np = mesh["u"]  

# Interpolation auf Raster für smooth Plot
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
Xi, Yi = np.meshgrid(xi, yi)
Ui = griddata((x, y), u_np, (Xi, Yi), method="linear")

# Plot
plt.figure(figsize=(7,5))
plt.contourf(Xi, Yi, Ui, levels=50, cmap="viridis")
plt.colorbar(label="u-velocity")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Interpolated u-velocity field for Re=360")
plt.gca().set_aspect("equal")
plt.show()


