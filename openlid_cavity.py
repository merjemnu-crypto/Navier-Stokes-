import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


#grid set to 64x64 for reasonable run time
nx, ny = 64, 64
Lx, Ly = 1.0, 1.0
dx, dy = Lx/nx, Ly/ny


#REs 
Re = 50            # Reynolds-Zahl
U = 1.0                  # Deckelgeschwindigkeit  
L = 1.0                  # Kastenlänge
nu = U * L / Re

dt = 0.08 * min(dx/U, dy/U, dx**2/nu, dy**2/nu)  # CFL based timestep

nt = 3000


# Fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))  # match the shape of u and v

# Lid-driven cavity boundary condition
def apply_bcs(u, v):
    u[0,:]  = 0
    u[-1,:] = 1   # top lid moves right
    u[:,0]  = 0
    u[:,-1] = 0
    v[0,:]  = 0
    v[-1,:] = 0
    v[:,0]  = 0
    v[:,-1] = 0
    return u, v
p[0,0] = 0   # Fix one corner to zero pressure

def pressure_poisson(p, rhs, dx, dy, max_iter=20000, tol=1e-8, omega=0.8, verbose=False):
    """
    - p   : initial guess (2D array, shape (ny, nx)).
    - rhs : right-hand side (2D array same shape as p). #in der projektion ist des div/dt
    - dx, dy : grid spacings in x and y.
    - max_iter, tol : stopping criteria based on L∞ residual.
    - omega : relaxation parameter (0 < omega < 1 for under-relaxation, ~0.7-0.9 often good).
    - verbose : print residual
    Returns the solution p 
    """
    # Precompute constant coefficients
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)   

    # Work arrays
    p_new = p.copy()           
    ny, nx = p.shape

 
    if rhs.shape != p.shape:
        raise ValueError("rhs and p must have same shape")

    # Initial residual 
    def compute_residual(P):
        lap = (P[1:-1,2:] - 2.0*P[1:-1,1:-1] + P[1:-1,:-2]) / dx2 + \
              (P[2:,1:-1] - 2.0*P[1:-1,1:-1] + P[:-2,1:-1]) / dy2
        res = np.max(np.abs(lap - rhs[1:-1,1:-1]))
        return res

    # Main iteration!!
    res = compute_residual(p)
    if verbose:
        print(f"[Poisson] initial residual = {res:.3e}")

    for it in range(1, max_iter+1):
        # Jacobi update (vectorized)
        p_new[1:-1,1:-1] = (
            (p[1:-1,2:] + p[1:-1,:-2]) * dy2 +
            (p[2:,1:-1] + p[:-2,1:-1]) * dx2 -
            rhs[1:-1,1:-1] * dx2 * dy2
        ) / denom

        # Weighted  Jacobi: combine old and new
        p[1:-1,1:-1] = (1.0 - omega) * p[1:-1,1:-1] + omega * p_new[1:-1,1:-1]

        # Neumann BCs (dp/dn = 0) 
        p[0, :]   = p[1, :]
        p[-1, :]  = p[-2, :]
        p[:, 0]   = p[:, 1]
        p[:, -1]  = p[:, -2]

        # Residual check every few iterations to avoid extra cost each loop
        if (it & 31) == 0:  # every 32 iterations
            res = compute_residual(p)
            if verbose:
                print(f"[Poisson] it={it}, residual={res:.3e}")
            if res < tol:
                if verbose:
                    print(f"[Poisson] converged after {it} iterations, residual={res:.3e}")
                break

    else:
        # reached max_iter without break
        res = compute_residual(p)
        if verbose:
            print(f"[Poisson] WARNING: max_iter reached ({max_iter}). final residual={res:.3e}")

    return p



div_history = [] # storing max div over time to keep track of convergence
# Time stepping
for n in range(nt):
    un, vn = u.copy(), v.copy()

    # tentative velocities u and v
    # Tentative velocity u (upwind for stability)
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
        dt/dx * ( (un[1:-1,1:-1] + np.abs(un[1:-1,1:-1]))/2 * (un[1:-1,1:-1] - un[1:-1,:-2]) +
                (un[1:-1,1:-1] - np.abs(un[1:-1,1:-1]))/2 * (un[1:-1,2:] - un[1:-1,1:-1]) ) -
        dt/dy * ( (vn[1:-1,1:-1] + np.abs(vn[1:-1,1:-1]))/2 * (un[1:-1,1:-1] - un[:-2,1:-1]) +
                (vn[1:-1,1:-1] - np.abs(vn[1:-1,1:-1]))/2 * (un[2:,1:-1] - un[1:-1,1:-1]) ) +
        nu*dt*( (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])/dx**2 +
                (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])/dy**2 ))
    # Tentative velocity v (upwind for stability)
    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
         dt/dx * ( (un[1:-1,1:-1] + np.abs(un[1:-1,1:-1]))/2 * (vn[1:-1,1:-1] - vn[1:-1,:-2]) +
              (un[1:-1,1:-1] - np.abs(un[1:-1,1:-1]))/2 * (vn[1:-1,2:] - vn[1:-1,1:-1]) ) -
          dt/dy * ( (vn[1:-1,1:-1] + np.abs(vn[1:-1,1:-1]))/2 * (vn[1:-1,1:-1] - vn[:-2,1:-1]) +
              (vn[1:-1,1:-1] - np.abs(vn[1:-1,1:-1]))/2 * (vn[2:,1:-1] - vn[1:-1,1:-1]) ) +
         nu*dt*( (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])/dx**2 +
            (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])/dy**2 ))





   
    # divergence (same shape as u, v, p)
    div = np.zeros_like(p)
    div[1:-1,1:-1] = ((u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                    (v[2:,1:-1] - v[:-2,1:-1])/(2*dy))
  


    p = pressure_poisson(p, div / dt, dx, dy, max_iter=20000, tol=1e-8, omega=0.8, verbose=False)

    # pressure gradients 
    dpdx = np.zeros_like(u)
    dpdy = np.zeros_like(v)

    dpdx[1:-1,1:-1] = (p[1:-1,2:] - p[1:-1,:-2]) / (2*dx)
    dpdy[1:-1,1:-1] = (p[2:,1:-1] - p[:-2,1:-1]) / (2*dy)

    # projection step
    u -= dt * dpdx
    v -= dt * dpdy
  

    # BCs enforcement AFTER projection step
    u, v = apply_bcs(u,v)
    
    # print max div every 50 steps
    if n % 50 == 0:
        print(f"step {n}, t={n*dt:.4f}, max|div|={np.abs(div).max():.3e}")
        div_history.append(np.abs(div).max())


# Visualization
X, Y = np.meshgrid(np.linspace(0,Lx,nx), np.linspace(0,Ly,ny))


# Vorticity w = dv/dx - du/dy
w = np.zeros((ny-2, nx-2))
w[:,:] = (v[1:-1,2:] - v[1:-1,:-2])/(2*dx) - (u[2:,1:-1] - u[:-2,1:-1])/(2*dy)

# Grid for vorticity 
Xc, Yc = np.meshgrid(np.linspace(dx, Lx-dx, nx-2), np.linspace(dy, Ly-dy, ny-2))

wmax = np.max(np.abs(w))

# Velocity magnitude 
vel_mag = np.sqrt(u**2 + v**2)



vmax=0.5
plt.figure(figsize=(6,5))
plt.pcolormesh(X, Y, vel_mag, cmap="viridis", shading="auto", vmin=0, vmax=vmax) # Farbskala besser symmetrisch von 0 bis max Geschwindigkeit
plt.colorbar(label="Velocity magnitude")
plt.streamplot(X, Y, u, v, color="k", density=1)
plt.title(f'2D Lid-Driven Cavity Flow for Re={Re}')
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()

#plot for max div over t 
plt.figure(figsize=(6,4))
plt.plot(np.arange(0, nt, 50), div_history, marker="o")
plt.yscale("log")  
plt.xlabel("n (timestep)")
plt.ylabel("max |div|")
plt.title("Convergence of the Divergence over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
