"""
plot_field.py — visualize point-source PML solution from pml_point_source_demo
Reads ParaView VTU output via pyvista, plots |E| on a 2D cross-section.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pyvista as pv
import os, glob

# ── locate latest VTU file ────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
vtu_files = sorted(glob.glob(f"{REPO}/ParaView/nurbs_ex25p/Cycle*/proc000000.vtu"))
if not vtu_files:
    raise FileNotFoundError("No VTU files found. Run pml_point_source_demo first.")
vtu_path = vtu_files[-1]
print(f"Reading: {vtu_path}")

# ── read with pyvista (handles all VTK element types) ────────────────────────
# merge all proc files if multiple ranks
cycle_dir = os.path.dirname(vtu_path)
pvtu = os.path.join(cycle_dir, "data.pvtu")
if not os.path.exists(pvtu):
    pvtu = vtu_path       # single rank fallback

mesh = pv.read(pvtu)
# for multi-block, get the first block
if hasattr(mesh, "combine"):
    mesh = mesh.combine()

print(f"Points: {mesh.n_points}  Cells: {mesh.n_cells}")
print(f"Point arrays: {mesh.point_data.keys()}")

pts = np.array(mesh.points)
x, y = pts[:, 0], pts[:, 1]

# ── identify field arrays ─────────────────────────────────────────────────────
keys = list(mesh.point_data.keys())
def pick(candidates):
    for c in candidates:
        for k in keys:
            if c.lower() in k.lower():
                return k
    return None

real_key = pick(["E_r","sol_r","real","Re","_r","Er"])
imag_key = pick(["E_i","sol_i","imag","Im","_i","Ei"])
print(f"  real → {real_key},   imag → {imag_key}")

def get_component0(key):
    if key is None:
        return None
    arr = np.array(mesh.point_data[key])
    return arr[:, 0] if arr.ndim == 2 else arr

Er0 = get_component0(real_key)
Ei0 = get_component0(imag_key)

# ── build Delaunay triangulation from point cloud ─────────────────────────────
triang = tri.Triangulation(x, y)

# ── plot ─────────────────────────────────────────────────────────────────────
ncols = 3 if (Er0 is not None and Ei0 is not None) else 1
fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 5.5))
if ncols == 1:
    axes = [axes]
fig.suptitle("Point-source Maxwell-PML  (IGA RAS-BiCGSTAB)", fontsize=13)

def triplot(ax, z, title, cmap="RdBu_r", symmetric=True):
    vmax = np.percentile(np.abs(z), 99)
    if symmetric:
        im = ax.tripcolor(triang, z, shading="gouraud",
                          cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.tripcolor(triang, z, shading="gouraud",
                          cmap="inferno", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")

if Er0 is not None and Ei0 is not None:
    amp = np.sqrt(Er0**2 + Ei0**2)
    triplot(axes[0], Er0, "Re(Ex) — point source + PML", cmap="RdBu_r")
    triplot(axes[1], Ei0, "Im(Ex)", cmap="RdBu_r")
    triplot(axes[2], amp, "|Ex| — field amplitude", symmetric=False)
elif Er0 is not None:
    triplot(axes[0], Er0, f"{real_key}", cmap="RdBu_r")
else:
    k0 = keys[0]
    d = get_component0(k0)
    triplot(axes[0], d, k0, cmap="RdBu_r")

plt.tight_layout()
out = f"{REPO}/field_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
