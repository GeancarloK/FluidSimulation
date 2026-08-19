
"""
visualize_density_interactive.py
 
Visualizador 2D interativo com:
  - Slider pra trocar a fatia (indice ao longo do eixo de corte)
  - RadioButtons pra trocar o plano de corte (XY, XZ, YZ)
  - Um unico vetor de velocidade RESULTANTE por celula (verde claro),
    partindo do CENTRO geometrico da celula — obtido promediando as
    velocidades de face da grade staggered (mesmo criterio usado em
    generate_slices_gif_pro.py).
 
Cola o texto do printf na variavel DATA e roda:
    py visualize_density_interactive.py
 
Requer: numpy, matplotlib
    py -m pip install numpy matplotlib
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import os
 
# ======================================================================
# Le o arquivo data.txt na mesma pasta do script
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "dataOpt_4194304_8192_512.txt")
 
with open(DATA_PATH, "r") as f:
    DATA = f.read()
# ======================================================================
 
 
HEADER_DIMS_RE = re.compile(
    r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"
)
HEADER_CELL_RE = re.compile(
    r"Thread size \(m\):\s*dxThreads=([\d.]+)\s+dyThreads=([\d.]+)\s+dzThreads=([\d.]+)"
)
HEADER_TIME_RE = re.compile(
    r"Total simulation time \(s\):\s*([\d.]+)"
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
    time_match = HEADER_TIME_RE.search(text)
    if not dims_match:
        raise ValueError("Nao encontrei 'Total threads: ...' no texto.")
    nx, ny, nz = (int(v) for v in dims_match.groups())
    dx, dy, dz = (float(v) for v in cell_match.groups()) if cell_match else (1.0, 1.0, 1.0)
    total_time = float(time_match.group(1)) if time_match else None
 
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
 
    return density, vel_x, vel_y, vel_z, cubos, (nx, ny, nz), (dx, dy, dz), total_time
 
 
def center_from_faces(face_h, face_v):
    """Promedia velocidades de face (grade staggered) para o CENTRO da celula.
 
    face_h[i, j]: velocidade na face -h da celula (i, j) (eixo horizontal).
    face_v[i, j]: velocidade na face -v da celula (i, j) (eixo vertical).
 
    Para a celula i, a componente horizontal no centro e' a media entre a
    face -h da propria celula (face_h[i]) e a face -h da celula seguinte
    (face_h[i+1], que fisicamente e' a face +h da celula i) — o mesmo se
    aplica ao eixo vertical. Nas bordas do dominio, usa-se so a face
    conhecida (nao ha face seguinte disponivel).
    """
    nh, nv = face_h.shape
 
    u_center = np.empty((nh, nv), dtype=np.float32)
    u_center[:-1, :] = 0.5 * (face_h[:-1, :] + face_h[1:, :])
    u_center[-1, :] = face_h[-1, :]
 
    w_center = np.empty((nh, nv), dtype=np.float32)
    w_center[:, :-1] = 0.5 * (face_v[:, :-1] + face_v[:, 1:])
    w_center[:, -1] = face_v[:, -1]
 
    return u_center, w_center
 
 
class InteractiveSliceViewer:
    def __init__(self, density, vel_x, vel_y, vel_z, cubos, dims, cell_size, total_time=None):
        self.density = density
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.cubos = cubos
        self.nx, self.ny, self.nz = dims
        self.dx, self.dy, self.dz = cell_size
        self.total_time = total_time
        self.vmin, self.vmax = density.min(), density.max()
        self.norm = Normalize(vmin=self.vmin, vmax=self.vmax)
 
        self.mode = "XY"   # plano de corte atual
        self.slice_idx = 0
 
        # --- layout da figura ---
        self.fig = plt.figure(figsize=(12, 8))
 
        # titulo geral da figura com o tempo de simulacao (se disponivel)
        if self.total_time is not None:
            self.fig.suptitle(f"Tempo total de simulacao: {self.total_time:.6f} s", fontsize=11)
 
        # eixo principal (plot)
        self.ax = self.fig.add_axes([0.1, 0.2, 0.65, 0.7])
 
        # slider de fatia
        ax_slider = self.fig.add_axes([0.1, 0.05, 0.65, 0.04])
        max_slices = self._max_slices()
        self.slider = Slider(ax_slider, "Fatia", 0, max(max_slices - 1, 0),
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)
 
        # radio buttons para plano
        ax_radio = self.fig.add_axes([0.82, 0.5, 0.15, 0.2])
        self.radio = RadioButtons(ax_radio, ["XY (varia z)", "XZ (varia y)", "YZ (varia x)"],
                                   active=0)
        self.radio.on_clicked(self._on_radio)
 
        # colorbar
        self.cbar_ax = self.fig.add_axes([0.82, 0.2, 0.03, 0.25])
 
        self._draw()
 
    def _max_slices(self):
        if self.mode == "XY":
            return self.nz
        elif self.mode == "XZ":
            return self.ny
        else:
            return self.nx
 
    def _on_radio(self, label):
        if "XY" in label:
            self.mode = "XY"
        elif "XZ" in label:
            self.mode = "XZ"
        else:
            self.mode = "YZ"
        self.slice_idx = 0
        ms = self._max_slices()
        self.slider.valmax = max(ms - 1, 0)
        self.slider.set_val(0)
        self.slider.ax.set_xlim(0, max(ms - 1, 0))
        self._draw()
 
    def _on_slider(self, val):
        self.slice_idx = int(val)
        self._draw()
 
    def _draw(self):
        self.ax.clear()
        self.cbar_ax.clear()
 
        idx = self.slice_idx
 
        if self.mode == "XY":
            # plano XY, cortando em z=idx
            # eixo horizontal = x, eixo vertical = y
            slice_dens = self.density[:, :, idx].T  # [ny, nx]
            # faces de velocidade no plano (mesma orientacao [nh, nv] dos
            # arrays vel_x/vel_y fatiados, usada pelo center_from_faces)
            face_h_slice = self.vel_x[:, :, idx]   # [nx, ny]
            face_v_slice = self.vel_y[:, :, idx]   # [nx, ny]
 
            h_size, v_size = self.dx, self.dy
            nh, nv = self.nx, self.ny
            h_label, v_label = "x (m)", "y (m)"
            title = f"Plano XY — z={idx} (z_pos={idx*self.dz:.3f}m)"
 
        elif self.mode == "XZ":
            # plano XZ, cortando em y=idx
            # eixo horizontal = x, eixo vertical = z
            slice_dens = self.density[:, idx, :].T  # [nz, nx]
            face_h_slice = self.vel_x[:, idx, :]   # [nx, nz]
            face_v_slice = self.vel_z[:, idx, :]   # [nx, nz]
 
            h_size, v_size = self.dx, self.dz
            nh, nv = self.nx, self.nz
            h_label, v_label = "x (m)", "z (m)"
            title = f"Plano XZ — y={idx} (y_pos={idx*self.dy:.3f}m)"
 
        else:  # YZ
            # plano YZ, cortando em x=idx
            # eixo horizontal = y, eixo vertical = z
            slice_dens = self.density[idx, :, :].T  # [nz, ny]
            face_h_slice = self.vel_y[idx, :, :]   # [ny, nz]
            face_v_slice = self.vel_z[idx, :, :]   # [ny, nz]
 
            h_size, v_size = self.dy, self.dz
            nh, nv = self.ny, self.nz
            h_label, v_label = "y (m)", "z (m)"
            title = f"Plano YZ — x={idx} (x_pos={idx*self.dx:.3f}m)"
 
        # heatmap
        extent = [0, nh * h_size, 0, nv * v_size]
        im = self.ax.imshow(slice_dens, origin="lower", extent=extent,
                            cmap="coolwarm", norm=self.norm, aspect="equal",
                            interpolation="nearest")
 
        # vetor resultante (verde claro) partindo do centro de cada celula:
        # promedia as faces -h/+h e -v/+v da grade staggered (mesmo
        # criterio de generate_slices_gif_pro.py)
        u_center, w_center = center_from_faces(face_h_slice, face_v_slice)
 
        centers_h = np.array([(i + 0.5) * h_size for i in range(nh)])
        centers_v = np.array([(j + 0.5) * v_size for j in range(nv)])
        C_h, C_v = np.meshgrid(centers_h, centers_v)   # [nv, nh]
 
        U = u_center.T   # [nv, nh]
        W = w_center.T   # [nv, nh]
 
        speed = np.sqrt(U**2 + W**2)
        max_speed = speed.max() if speed.max() > 0 else 1.0
        arrow_scale = max_speed * 15
 
        self.ax.quiver(C_h, C_v, U, W,
                       color="lightgreen", alpha=0.85,
                       scale=arrow_scale,
                       width=0.0012, headwidth=2.5, headlength=3.5,
                       label="velocidade resultante")
 
        # cubos pretos: quadrado centrado na origem (ix*dx, iy*dy) da celula
        if self.mode == "XY":
            cubos_slice = self.cubos[:, :, idx]  # [nx, ny]
            for ix in range(self.nx):
                for iy in range(self.ny):
                    if cubos_slice[ix, iy]:
                        cx, cy = ix * self.dx, iy * self.dy
                        rect = Rectangle(
                            (cx - h_size/2, cy - v_size/2),
                            h_size, v_size,
                            facecolor="black", edgecolor="black", alpha=0.85
                        )
                        self.ax.add_patch(rect)
        elif self.mode == "XZ":
            cubos_slice = self.cubos[:, idx, :]  # [nx, nz]
            for ix in range(self.nx):
                for iz in range(self.nz):
                    if cubos_slice[ix, iz]:
                        cx, cz = ix * self.dx, iz * self.dz
                        rect = Rectangle(
                            (cx - h_size/2, cz - v_size/2),
                            h_size, v_size,
                            facecolor="black", edgecolor="black", alpha=0.85
                        )
                        self.ax.add_patch(rect)
        else:  # YZ
            cubos_slice = self.cubos[idx, :, :]  # [ny, nz]
            for iy in range(self.ny):
                for iz in range(self.nz):
                    if cubos_slice[iy, iz]:
                        cy, cz = iy * self.dy, iz * self.dz
                        rect = Rectangle(
                            (cy - h_size/2, cz - v_size/2),
                            h_size, v_size,
                            facecolor="black", edgecolor="black", alpha=0.85
                        )
                        self.ax.add_patch(rect)
 
        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel(h_label)
        self.ax.set_ylabel(v_label)
        self.ax.legend(loc="upper right", fontsize=8)
 
        self.fig.colorbar(im, cax=self.cbar_ax, label="densidade (kg/m³)")
        self.fig.canvas.draw_idle()
 
    def show(self):
        plt.show()
 
 
if __name__ == "__main__":
 
    density, vx, vy, vz, cubos, dims, cell_size, total_time = parse_data(DATA)
    print(f"Grid: {dims[0]}x{dims[1]}x{dims[2]}, celula = {cell_size}")
    print(f"Densidade min={density.min():.6f} max={density.max():.6f}")
    print(f"|Vel| max = {np.sqrt(vx**2 + vy**2 + vz**2).max():.4f}")
    print(f"Cubos ativos: {cubos.sum()} / {cubos.size}")
    if total_time is not None:
        print(f"Tempo total de simulacao: {total_time:.6f} s")
 
    viewer = InteractiveSliceViewer(density, vx, vy, vz, cubos, dims, cell_size, total_time)
    viewer.show()
 