import numpy as np
import matplotlib.pyplot as plt
from derivatives import Diff1, Diff2
from scipy import sparse
from fft_poisson import fft_poisson

# Physical parameters
Re = 70
Lx, Ly = 1, 1

# Numerical parameters
nx, ny = 20, 20
nu = 1 / Re
tf = 10
max_co = 1
order = 4
dt = 0.002
# Boundary conditions
u0, v0 = 0, 0
u1, u2, u3, u4 = 0, 1, 0, 0
v1, v2, v3, v4 = 0, 0, 0, 0

# Generate mesh
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
dx, dy = x[1]-x[0], y[1]-y[0]

# Operators
d_x = Diff1(nx, order) / dx
d_y = Diff1(ny, order) / dy
d_x2 = Diff2(nx, order) / dx**2
d_y2 = Diff2(ny, order) / dy**2
I = np.eye(nx, ny)
DX = sparse.kron(d_x, I)
DY = sparse.kron(I, d_y)
DX2 = sparse.kron(d_x2, I)
DY2 = sparse.kron(I, d_y2)

# Iterations
it_max = int(tf/dt) - 1


r1 = u1*dt/dx
r2 = u1*dt/dy
if r1 > max_co or r2 > max_co:
    raise RuntimeError("Unstable solution (CFL too large)")


# -------------------- Initialize fields in a dictionary --------------------
fields = {name: np.zeros((nx, ny)) for name in
          ["u", "v", "w", "psi", "p",
           "dwdx", "dwdy", "d2wdx2", "d2wdy2",
           "dpsidx", "dpsidy", "dudx", "dudy", "dvdx", "dvdy"]}

# Initial condition
fields["u"].fill(u0)
fields["v"].fill(v0)

for t in range(it_max):
    u, v, w, psi, p = fields["u"], fields["v"], fields["w"], fields["psi"], fields["p"]

    # Apply BCs
    u[0, :], u[-1, :], v[0, :], v[-1, :] = u3, u4, v3, v4
    u[:, 0], u[:, -1], v[:, 0], v[:, -1] = u1, u2, v1, v2

    # Vorticity boundaries
    dudy = np.reshape(DY @ u.flatten(), (nx, ny))
    dvdx = np.reshape(DX @ v.flatten(), (nx, ny))
    w[0, :], w[-1, :] = dvdx[0, :] - dudy[0, :], dvdx[-1, :] - dudy[-1, :]
    w[:, 0], w[:, -1] = dvdx[:, 0] - dudy[:, 0], dvdx[:, -1] - dudy[:, -1]

    # Streamfunction
    psi = fft_poisson(-w, dx)
    fields["psi"] = psi

    # Derivatives of vorticity
    fields["dwdx"] = np.reshape(DX @ w.flatten(), (nx, ny))
    fields["dwdy"] = np.reshape(DY @ w.flatten(), (nx, ny))
    fields["d2wdx2"] = np.reshape(DX2 @ w.flatten(), (nx, ny))
    fields["d2wdy2"] = np.reshape(DY2 @ w.flatten(), (nx, ny))

    # Time advancement
    w += dt * (-u*fields["dwdx"] - v*fields["dwdy"] + (1/Re)*(fields["d2wdx2"] + fields["d2wdy2"]))
    fields["w"] = w

    # Update streamfunction and velocities
    psi = fft_poisson(-w, dx)
    dpsidx = np.reshape(DX @ psi.flatten(), (nx, ny))
    dpsidy = np.reshape(DY @ psi.flatten(), (nx, ny))
    fields["u"], fields["v"] = dpsidy, -dpsidx
    u, v = fields["u"], fields["v"]

    # Continuity check
    dudx = np.reshape(DX @ u.flatten(), (nx, ny))
    dvdy = np.reshape(DY @ v.flatten(), (nx, ny))
    continuity = dudx + dvdy
    print(f"Iteration {t}: continuity max {continuity.max():.3e}, min {continuity.min():.3e}")

    # Pressure
    dudy = np.reshape(DY @ u.flatten(), (nx, ny))
    dvdx = np.reshape(DX @ v.flatten(), (nx, ny))
    f = dudx**2 + dvdy**2 + 2*dudy*dvdx
    p = fft_poisson(-f, dx)
    fields["p"] = p

    if np.isclose(t*dt, 0.02) or np.isclose(t*dt, 5) or np.isclose(t*dt, 9):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        # Streamfunction plot
        contour = axs[0].contourf(X, Y, psi, 40, cmap='viridis')
        axs[0].set_title(f'Streamfunction at t = {t*dt:.2f} s')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_aspect('equal', adjustable='box')
        plt.colorbar(contour, ax=axs[0])

        # Velocity quiver plot
        axs[1].quiver(X, Y, u, v, cmap='gist_rainbow_r', alpha=0.8, scale=50)
        axs[1].set_title(f'Velocity Field at t = {t*dt:.2f} s')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_aspect('equal')

        plt.tight_layout()
        plt.show()
        plt.close()
