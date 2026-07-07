"""
generate_slices_gif.py

Gera GIFs animados das fatias XY (variando z) e YZ (variando x).

Uso:
    py generate_slices_gif.py

Requer: numpy, matplotlib, pillow
    py -m pip install numpy matplotlib pillow
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from PIL import Image

# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data_8192_1024.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "frames")
# ======================================================================

HEADER_DIMS_RE = re.compile(
    r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"
)
HEADER_CELL_RE = re.compile(
    r"Thread size \(m\):\s*dxThreads=([\d.]+)\s+dyThreads=([\d.]+)\s+dzThreads=([\d.]+)"
)
ROW_RE = re.compile(
    r"\[(\d+)\]\s*\(x=(\d+)\s+y=(\d+)\s+z=(\d+)\)\s+"
    r"mass=([-\d.]+)\s+volume=([-\d.]+)\s+density=([-\d.]+)\s+"
    r"cubos=(\d+)\s*"
    r"xArea=([-\d.]+)\s+yArea=([-\d.]+)\s+zArea=([-\d.]+)\s+"
    r"xVel=([-\d.]+)\s+yVel=([-\d.]+)\s+zVel=([-\d.]+)"
)


def parse_data(text):
    dims_match = HEADER_DIMS_RE.search(text)
    cell_match = HEADER_CELL_RE.search(text)
    if not dims_match:
        raise ValueError("Nao encontrei 'Total threads: ...' no texto.")
    nx, ny, nz = (int(v) for v in dims_match.groups())
    dx, dy, dz = (float(v) for v in cell_match.groups()) if cell_match else (1.0, 1.0, 1.0)

    density = np.zeros((nx, ny, nz), dtype=np.float32)
    vel_x   = np.zeros((nx, ny, nz), dtype=np.float32)
    vel_y   = np.zeros((nx, ny, nz), dtype=np.float32)
    vel_z   = np.zeros((nx, ny, nz), dtype=np.float32)
    cubos   = np.zeros((nx, ny, nz), dtype=np.bool_)

    for m in ROW_RE.finditer(text):
        (_k, x, y, z, _mass, _vol, dens, cubo,
         _xa, _ya, _za, xv, yv, zv) = m.groups()
        x, y, z = int(x), int(y), int(z)
        density[x, y, z] = float(dens)
        cubos[x, y, z]   = int(cubo) != 0
        vel_x[x, y, z]   = float(xv)
        vel_y[x, y, z]   = float(yv)
        vel_z[x, y, z]   = float(zv)

    return density, vel_x, vel_y, vel_z, cubos, (nx, ny, nz), (dx, dy, dz)


def render_slice(density_2d, cubos_2d, vx_2d, vy_2d,
                 h_size, v_size, nh, nv,
                 h_label, v_label, title, norm, filepath):
    """Renderiza uma fatia e salva como PNG."""
    fig, ax = plt.subplots(figsize=(12, 8))

    extent = [0, nh * h_size, 0, nv * v_size]
    ax.imshow(density_2d, origin="lower", extent=extent,
              cmap="coolwarm", norm=norm, aspect="equal",
              interpolation="nearest")

    # setas de velocidade
    all_speeds = np.concatenate([np.abs(vx_2d.ravel()), np.abs(vy_2d.ravel())])
    max_speed = all_speeds.max() if all_speeds.max() > 0 else 1.0
    arrow_scale = max_speed * 15

    # posições das setas horizontais
    hx = np.array([i * h_size for i in range(nh)])
    hy = np.array([j * v_size + v_size / 2 for j in range(nv)])
    Hx, Hy = np.meshgrid(hx, hy)
    ax.quiver(Hx, Hy, vx_2d, np.zeros_like(vx_2d),
              color="blue", alpha=0.8, scale=arrow_scale,
              width=0.003, headwidth=3, headlength=4)

    # posições das setas verticais
    vxp = np.array([i * h_size + h_size / 2 for i in range(nh)])
    vyp = np.array([j * v_size for j in range(nv)])
    Vx, Vy = np.meshgrid(vxp, vyp)
    ax.quiver(Vx, Vy, np.zeros_like(vy_2d), vy_2d,
              color="green", alpha=0.8, scale=arrow_scale,
              width=0.003, headwidth=3, headlength=4)

    # cubos pretos
    for i in range(cubos_2d.shape[1]):  # nh
        for j in range(cubos_2d.shape[0]):  # nv
            if cubos_2d[j, i]:
                rect = Rectangle(
                    (i * h_size - h_size / 2, j * v_size - v_size / 2),
                    h_size, v_size,
                    facecolor="black", edgecolor="black", alpha=0.85
                )
                ax.add_patch(rect)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)

    fig.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_dir, prefix, output_path, duration_ms=200):
    """Junta PNGs em GIF animado."""
    files = sorted(
        [f for f in os.listdir(frame_dir) if f.startswith(prefix) and f.endswith(".png")],
        key=lambda f: int(re.search(r"(\d+)", f.replace(prefix, "")).group())
    )
    if not files:
        print(f"Nenhum frame encontrado com prefixo '{prefix}'")
        return

    frames = [Image.open(os.path.join(frame_dir, f)) for f in files]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"GIF salvo: {output_path} ({len(frames)} frames)")


def main():
    with open(DATA_PATH, "r") as f:
        text = f.read()

    density, vx, vy, vz, cubos, (nx, ny, nz), (dx, dy, dz) = parse_data(text)
    norm = Normalize(vmin=density.min(), vmax=density.max())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- XY (variando z) ---
    print(f"Gerando {nz} fatias XY...")
    for iz in range(nz):
        render_slice(
            density_2d=density[:, :, iz].T,
            cubos_2d=cubos[:, :, iz].T,
            vx_2d=vx[:, :, iz].T,
            vy_2d=vy[:, :, iz].T,
            h_size=dx, v_size=dy, nh=nx, nv=ny,
            h_label="x (m)", v_label="y (m)",
            title=f"Plano XY — z={iz} (z_pos={iz*dz:.3f}m)",
            norm=norm,
            filepath=os.path.join(OUTPUT_DIR, f"xy_{iz:04d}.png")
        )
        print(f"  XY {iz+1}/{nz}", end="\r")
    print()

    # --- YZ (variando x) ---
    print(f"Gerando {nx} fatias YZ...")
    for ix in range(nx):
        render_slice(
            density_2d=density[ix, :, :].T,
            cubos_2d=cubos[ix, :, :].T,
            vx_2d=vy[ix, :, :].T,
            vy_2d=vz[ix, :, :].T,
            h_size=dy, v_size=dz, nh=ny, nv=nz,
            h_label="y (m)", v_label="z (m)",
            title=f"Plano YZ — x={ix} (x_pos={ix*dx:.3f}m)",
            norm=norm,
            filepath=os.path.join(OUTPUT_DIR, f"yz_{ix:04d}.png")
        )
        print(f"  YZ {ix+1}/{nx}", end="\r")
    print()

    # --- Gera GIFs ---
    make_gif(OUTPUT_DIR, "xy_", os.path.join(SCRIPT_DIR, "slices_xy.gif"), duration_ms=150)
    make_gif(OUTPUT_DIR, "yz_", os.path.join(SCRIPT_DIR, "slices_yz.gif"), duration_ms=150)

    print("Pronto!")


if __name__ == "__main__":
    main()