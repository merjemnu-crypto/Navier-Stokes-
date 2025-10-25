import os
import pyvista as pv
import numpy as np

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
vtu_path = os.path.join(script_dir, 'flowfield_Re160.vtu')  
mesh = pv.read(vtu_path)

# --- 3D velocity vectors from 2D ---
vectors = np.zeros((mesh.n_points, 3))
vectors[:, 0] = mesh['u']
vectors[:, 1] = mesh['v']
vectors[:, 2] = 0
mesh['velocity'] = vectors

# --- Seed points for streamlines ---
x = np.linspace(mesh.bounds[0]+0.1, mesh.bounds[1]-0.1, 20)  # fewer seeds
y = np.linspace(mesh.bounds[2]+0.1, mesh.bounds[3]-0.1, 10)
seeds_points = np.array([[xi, yi, 0] for xi in x for yi in y])
seeds = pv.PolyData(seeds_points)

# --- Compute streamlines ---
streamlines = mesh.streamlines_from_source(
    source=seeds,
    vectors='velocity',
    integrator_type=45,  # Runge-Kutta 4/5
    max_time=20.0,
    initial_step_length=0.05,
    terminal_speed=1e-5
)

# --- Plotting ---
p = pv.Plotter(window_size=[1200, 800])
p.add_mesh(mesh, scalars='p', cmap='coolwarm', opacity=0.5, show_edges=False)

# Plot streamlines as normal lines (not tubes)
p.add_mesh(streamlines, color='white', line_width=1)  # line_width controls thickness

# Optional: arrows along the velocity field (optional)
arrows = mesh.glyph(orient='velocity', scale='velocity', factor=0.05)
p.add_mesh(arrows, color='yellow')

p.show_grid()
p.show()
