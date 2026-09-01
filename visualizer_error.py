"""
visualizer_thresholds.py

Variante de visualizer.py: em vez de pintar o degrade continuo de
densidades (colormap), este visualizador pinta APENAS celulas fora de
uma faixa de densidade:

  - Densidade < 0            -> quadrado AZUL
  - Densidade > 103          -> quadrado VERMELHO

Os quadrados azul/vermelho ficam ACIMA dos quadrados PRETOS (cubos),
mas ABAIXO dos quadrados LARANJA (warpInfo != 0) — ou seja, se uma
celula tiver warpInfo != 0, o laranja continua tendo prioridade visual
sobre azul/vermelho, do mesmo jeito que ja tinha prioridade sobre o
preto no visualizer.py original.

Todo o resto (slider de fatia, RadioButtons de plano, vetor de
velocidade resultante partindo do centro da celula, linhas amarelas de
bloco CUDA) e' identico ao visualizer.py.

Uso:
    python3 visualizer_thresholds.py <numBlocks> <numThreads>

O arquivo informado e' procurado dentro da pasta 'data', na mesma pasta
deste script (ex.: data/dataOpt_524288_1024_512.txt).

Requer: numpy, matplotlib
    py -m pip install numpy matplotlib
"""
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.patches import Rectangle
import os

# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Limites de densidade fora dos quais a celula e' pintada.
DENSITY_LOW_THRESHOLD = 0.0     # abaixo disso -> azul
DENSITY_HIGH_THRESHOLD = 103.0  # acima disso  -> vermelho
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
HEADER_GENCUBES_TIME_RE = re.compile(
    r"generateCubes time \(s\):\s*([\d.]+)"
)
HEADER_TOTAL_THREADS_RE = re.compile(
    r"totalThreads=(\d+)"
)
HEADER_CUBES_INFO_RE = re.compile(
    r"Cubes Info:\s*numCubes=(\d+)\s+occupiedVolume=([\d.]+)%\s+skippedWarps=([\d.]+)%"
)
# Dimensao do bloco CUDA em numero de celulas/threads por eixo. Ja vem
# no cabecalho do arquivo, na linha:
#   Threads per block: nxThreads=<int>  nyThreads=<int>  nzThreads=<int>
HEADER_BLOCKDIM_RE = re.compile(
    r"Threads per block:\s*nxThreads=(\d+)\s+nyThreads=(\d+)\s+nzThreads=(\d+)"
)
# Linha de dados: note que "warpskip %d" e' seguido IMEDIATAMENTE por
# "xArea=" sem separador (assim como o fprintf original concatena os
# literais), entao o \d+ do warpInfo para naturalmente antes do "xArea".
ROW_RE = re.compile(
    r"\[(\d+)\]\s*\(x=(\d+)\s+y=(\d+)\s+z=(\d+)\)\s+"
    r"mass=([-\d.]+)\s+volume=([-\d.]+)\s+density=([-\d.]+)\s+"
    r"cubos=(\d+)\s+warpskip[=\s]+(\d+)\s*"
    r"xArea=([-\d.]+)\s+yArea=([-\d.]+)\s+zArea=([-\d.]+)\s+"
    r"xVel=([-\d.]+)\s+yVel=([-\d.]+)\s+zVel=([-\d.]+)"
)


def resolve_data_path(argv):
    """Valida os argumentos de linha de comando e resolve o caminho do
    arquivo de dados dentro da pasta DATA_DIR.

    Encerra o programa (sys.exit) com uma mensagem de erro clara quando:
      - a quantidade de argumentos esta errada;
      - o arquivo informado nao existe dentro de DATA_DIR.
    """
    prog = os.path.basename(argv[0])

    if len(argv) != 3:
        print(
            f"Uso incorreto de argumentos ({len(argv) - 1} fornecido(s)).\n\n"
            f"Forma de chamada esperada:\n"
            f"  py {prog} <numBlocks> <numThreads>\n\n"
            f"O arquivo deve estar dentro da pasta:\n"
            f"  {DATA_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        numBlocks = int(argv[1])
        numThreads = int(argv[2])
    except ValueError:
        print(f"Erro: numBlocks e numThreads devem ser numeros inteiros. Recebido: numBlocks='{argv[1]}' numThreads='{argv[2]}'.", file=sys.stderr)
        sys.exit(1)
    total_threads = numBlocks * numThreads
    data_filename = f"dataOpt_{total_threads}_{numBlocks}_{numThreads}.txt"
    data_path = os.path.join(DATA_DIR, data_filename)

    if not os.path.isfile(data_path):
        print(
            f"Erro: arquivo nao encontrado: {data_path}\n"
            f"Verifique se '{data_filename}' esta dentro da pasta '{DATA_DIR}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return data_path


def parse_data(text):
    dims_match = HEADER_DIMS_RE.search(text)
    cell_match = HEADER_CELL_RE.search(text)
    time_match = HEADER_TIME_RE.search(text)
    gencubes_match = HEADER_GENCUBES_TIME_RE.search(text)
    total_threads_match = HEADER_TOTAL_THREADS_RE.search(text)
    cubes_info_match = HEADER_CUBES_INFO_RE.search(text)
    blockdim_match = HEADER_BLOCKDIM_RE.search(text)
    if not dims_match:
        raise ValueError("Nao encontrei 'Total threads: ...' no texto.")
    nx, ny, nz = (int(v) for v in dims_match.groups())
    dx, dy, dz = (float(v) for v in cell_match.groups()) if cell_match else (1.0, 1.0, 1.0)
    total_time = float(time_match.group(1)) if time_match else None
    gencubes_time = float(gencubes_match.group(1)) if gencubes_match else None
    total_threads_declared = int(total_threads_match.group(1)) if total_threads_match else None

    if cubes_info_match:
        num_cubes = int(cubes_info_match.group(1))
        occupied_volume_pct = float(cubes_info_match.group(2))
        skipped_warps_pct = float(cubes_info_match.group(3))
    else:
        num_cubes = None
        occupied_volume_pct = None
        skipped_warps_pct = None

    if blockdim_match:
        block_dims = tuple(int(v) for v in blockdim_match.groups())
    else:
        block_dims = None

    density = np.zeros((nx, ny, nz), dtype=np.float32)
    vel_x   = np.zeros((nx, ny, nz), dtype=np.float32)
    vel_y   = np.zeros((nx, ny, nz), dtype=np.float32)
    vel_z   = np.zeros((nx, ny, nz), dtype=np.float32)
    cubos   = np.zeros((nx, ny, nz), dtype=np.bool_)
    warp_info = np.zeros((nx, ny, nz), dtype=np.bool_)

    for m in ROW_RE.finditer(text):
        (_k, x, y, z, _mass, _vol, dens, cubo, warp,
         _xa, _ya, _za, xv, yv, zv) = m.groups()
        x, y, z = int(x), int(y), int(z)
        density[x, y, z]   = float(dens)
        cubos[x, y, z]     = int(cubo) != 0
        warp_info[x, y, z] = int(warp) != 0
        vel_x[x, y, z]     = float(xv)
        vel_y[x, y, z]     = float(yv)
        vel_z[x, y, z]     = float(zv)

    return {
        "density": density,
        "vel_x": vel_x,
        "vel_y": vel_y,
        "vel_z": vel_z,
        "cubos": cubos,
        "warp_info": warp_info,
        "dims": (nx, ny, nz),
        "cell_size": (dx, dy, dz),
        "block_dims": block_dims,
        "total_time": total_time,
        "gencubes_time": gencubes_time,
        "total_threads_declared": total_threads_declared,
        "num_cubes": num_cubes,
        "occupied_volume_pct": occupied_volume_pct,
        "skipped_warps_pct": skipped_warps_pct,
    }


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


def draw_block_grid(ax, block_dim_h, block_dim_v, h_size, v_size, nh, nv):
    """Desenha linhas amarelas fechando um quadrado a cada bloco CUDA.

    block_dim_h / block_dim_v: numero de celulas por bloco no eixo
    horizontal/vertical do plano atual (None ou <=0 desativa a linha
    correspondente).
    """
    if block_dim_h and block_dim_h > 0:
        xs = list(range(0, nh, block_dim_h))
        if xs[-1] != nh:
            xs.append(nh)
        for i in xs:
            ax.axvline(i * h_size, color="yellow", linewidth=0.6,
                       alpha=0.7, zorder=6)

    if block_dim_v and block_dim_v > 0:
        ys = list(range(0, nv, block_dim_v))
        if ys[-1] != nv:
            ys.append(nv)
        for j in ys:
            ax.axhline(j * v_size, color="yellow", linewidth=0.6,
                       alpha=0.7, zorder=6)


class InteractiveSliceViewer:
    def __init__(self, density, vel_x, vel_y, vel_z, cubos, warp_info, dims,
                 cell_size, block_dims=None,
                 total_time=None, gencubes_time=None,
                 num_cubes=None, occupied_volume_pct=None, skipped_warps_pct=None):
        self.density = density
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.cubos = cubos
        self.warp_info = warp_info
        self.nx, self.ny, self.nz = dims
        self.dx, self.dy, self.dz = cell_size
        self.block_dims = block_dims  # (xBlockDim, yBlockDim, zBlockDim) ou None
        self.total_time = total_time
        self.gencubes_time = gencubes_time
        self.num_cubes = num_cubes
        self.occupied_volume_pct = occupied_volume_pct
        self.skipped_warps_pct = skipped_warps_pct

        self.mode = "XY"   # plano de corte atual
        self.slice_idx = 0

        # --- layout da figura ---
        self.fig = plt.figure(figsize=(12, 8))

        # titulo geral da figura com o tempo de simulacao e estatisticas
        # de cubos/warps (quando disponiveis)
        title_bits = []
        if self.total_time is not None:
            title_bits.append(f"Tempo total de simulacao: {self.total_time:.6f} s")
        if self.gencubes_time is not None:
            title_bits.append(f"generateCubes: {self.gencubes_time:.6f} s")
        if self.num_cubes is not None:
            title_bits.append(
                f"cubos={self.num_cubes} ocupado={self.occupied_volume_pct:.2f}% "
                f"warpsPulados={self.skipped_warps_pct:.2f}%"
            )
        if title_bits:
            self.fig.suptitle("  |  ".join(title_bits), fontsize=11)

        # eixo principal (plot) — sem colorbar de densidade, entao a
        # area do plot pode ocupar mais espaco horizontal
        self.ax = self.fig.add_axes([0.1, 0.2, 0.75, 0.7])

        # slider de fatia
        ax_slider = self.fig.add_axes([0.1, 0.05, 0.65, 0.04])
        max_slices = self._max_slices()
        self.slider = Slider(ax_slider, "Fatia", 0, max(max_slices - 1, 0),
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        # radio buttons para plano
        ax_radio = self.fig.add_axes([0.86, 0.5, 0.12, 0.2])
        self.radio = RadioButtons(ax_radio, ["XY (varia z)", "XZ (varia y)", "YZ (varia x)"],
                                   active=0)
        self.radio.on_clicked(self._on_radio)

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

    def _block_dims_for_plane(self):
        """Retorna (block_dim_h, block_dim_v) para o plano/modo atual, ou
        (None, None) se nao houver info de bloco disponivel."""
        if self.block_dims is None:
            return None, None
        xb, yb, zb = self.block_dims
        if self.mode == "XY":
            return xb, yb
        elif self.mode == "XZ":
            return xb, zb
        else:  # YZ
            return yb, zb

    def _draw(self):
        self.ax.clear()

        idx = self.slice_idx

        if self.mode == "XY":
            # plano XY, cortando em z=idx
            # eixo horizontal = x, eixo vertical = y
            dens_slice = self.density[:, :, idx]   # [nx, ny]
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
            dens_slice = self.density[:, idx, :]   # [nx, nz]
            face_h_slice = self.vel_x[:, idx, :]   # [nx, nz]
            face_v_slice = self.vel_z[:, idx, :]   # [nx, nz]

            h_size, v_size = self.dx, self.dz
            nh, nv = self.nx, self.nz
            h_label, v_label = "x (m)", "z (m)"
            title = f"Plano XZ — y={idx} (y_pos={idx*self.dy:.3f}m)"

        else:  # YZ
            # plano YZ, cortando em x=idx
            # eixo horizontal = y, eixo vertical = z
            dens_slice = self.density[idx, :, :]   # [ny, nz]
            face_h_slice = self.vel_y[idx, :, :]   # [ny, nz]
            face_v_slice = self.vel_z[idx, :, :]   # [ny, nz]

            h_size, v_size = self.dy, self.dz
            nh, nv = self.ny, self.nz
            h_label, v_label = "y (m)", "z (m)"
            title = f"Plano YZ — x={idx} (x_pos={idx*self.dx:.3f}m)"

        extent = [0, nh * h_size, 0, nv * v_size]

        # fundo neutro (sem degrade de densidade): apenas define os
        # limites/aspecto do plot, ja que nao ha mais imshow de densidade
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("white")

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
                       zorder=7,
                       label="velocidade resultante")

        # cubos / densidade fora da faixa / warp: tres passes separados,
        # do mais baixo para o mais alto na pilha de z-order:
        #
        #  1) preto (cubos) — quadrado centrado no PONTO DE ORIGEM da
        #     celula (ia*size_a, ib*size_b), como no visualizer.py
        #     original.
        #  2) azul/vermelho (densidade fora da faixa [0, 103]) — desenhado
        #     por CIMA do preto, sobre a propria celula (intervalo
        #     [ia*size_a, (ia+1)*size_a) x [ib*size_b, (ib+1)*size_b)),
        #     acompanhando o mesmo criterio usado pelo laranja.
        #  3) laranja (warpInfo) — continua por cima de tudo, mantendo a
        #     prioridade visual que ja tinha no visualizer.py original.
        if self.mode == "XY":
            cubos_slice = self.cubos[:, :, idx]      # [nx, ny]
            warp_slice = self.warp_info[:, :, idx]   # [nx, ny]
            axis_a, axis_b, size_a, size_b = self.nx, self.ny, h_size, v_size
        elif self.mode == "XZ":
            cubos_slice = self.cubos[:, idx, :]      # [nx, nz]
            warp_slice = self.warp_info[:, idx, :]   # [nx, nz]
            axis_a, axis_b, size_a, size_b = self.nx, self.nz, h_size, v_size
        else:  # YZ
            cubos_slice = self.cubos[idx, :, :]      # [ny, nz]
            warp_slice = self.warp_info[idx, :, :]   # [ny, nz]
            axis_a, axis_b, size_a, size_b = self.ny, self.nz, h_size, v_size

        # dens_slice esta na mesma orientacao [axis_a, axis_b] que
        # cubos_slice/warp_slice (ambos vem do mesmo corte [nx/ny/nz, ...]),
        # entao dens_slice[ia, ib] corresponde a celula (ia, ib).

        # 1) preto (cubos)
        for ia in range(axis_a):
            for ib in range(axis_b):
                if not cubos_slice[ia, ib]:
                    continue
                ca, cb = ia * size_a, ib * size_b
                rect = Rectangle(
                    (ca - size_a / 2, cb - size_b / 2),
                    size_a, size_b,
                    facecolor="black", edgecolor="black", alpha=0.85,
                    zorder=2,
                )
                self.ax.add_patch(rect)

        # 2) azul/vermelho (densidade fora da faixa), acima do preto
        for ia in range(axis_a):
            for ib in range(axis_b):
                d = dens_slice[ia, ib]
                if d < DENSITY_LOW_THRESHOLD:
                    color = "blue"
                elif d > DENSITY_HIGH_THRESHOLD:
                    color = "red"
                else:
                    continue
                rect = Rectangle(
                    (ia * size_a, ib * size_b),
                    size_a, size_b,
                    facecolor=color, edgecolor=color, alpha=0.85,
                    zorder=3,
                )
                self.ax.add_patch(rect)

        # 3) laranja (warpInfo), acima de tudo
        for ia in range(axis_a):
            for ib in range(axis_b):
                if not warp_slice[ia, ib]:
                    continue
                rect = Rectangle(
                    (ia * size_a, ib * size_b),
                    size_a, size_b,
                    facecolor="orange", edgecolor="orange", alpha=0.85,
                    zorder=4,
                )
                self.ax.add_patch(rect)

        # quadrados amarelos delimitando cada bloco CUDA
        block_dim_h, block_dim_v = self._block_dims_for_plane()
        draw_block_grid(self.ax, block_dim_h, block_dim_v, h_size, v_size, nh, nv)

        self.ax.set_title(title, fontsize=12)
        self.ax.set_xlabel(h_label)
        self.ax.set_ylabel(v_label)

        # legenda manual (sem colorbar de densidade)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="lightgreen", lw=2, label="velocidade resultante"),
            Patch(facecolor="black", edgecolor="black", label="cubo"),
            Patch(facecolor="blue", edgecolor="blue",
                  label=f"densidade < {DENSITY_LOW_THRESHOLD:g}"),
            Patch(facecolor="red", edgecolor="red",
                  label=f"densidade > {DENSITY_HIGH_THRESHOLD:g}"),
            Patch(facecolor="orange", edgecolor="orange", label="warp pulado"),
        ]
        self.ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":

    data_path = resolve_data_path(sys.argv)
    with open(data_path, "r") as f:
        DATA = f.read()

    parsed = parse_data(DATA)
    density = parsed["density"]
    vx, vy, vz = parsed["vel_x"], parsed["vel_y"], parsed["vel_z"]
    cubos = parsed["cubos"]
    warp_info = parsed["warp_info"]
    dims = parsed["dims"]
    cell_size = parsed["cell_size"]
    block_dims = parsed["block_dims"]
    total_time = parsed["total_time"]
    gencubes_time = parsed["gencubes_time"]
    total_threads_declared = parsed["total_threads_declared"]
    num_cubes = parsed["num_cubes"]
    occupied_volume_pct = parsed["occupied_volume_pct"]
    skipped_warps_pct = parsed["skipped_warps_pct"]

    print(f"Arquivo: {data_path}")
    print(f"Grid: {dims[0]}x{dims[1]}x{dims[2]}, celula = {cell_size}")
    print(f"Densidade min={density.min():.6f} max={density.max():.6f}")
    print(f"Celulas com densidade < {DENSITY_LOW_THRESHOLD:g}: {int((density < DENSITY_LOW_THRESHOLD).sum())} / {density.size}")
    print(f"Celulas com densidade > {DENSITY_HIGH_THRESHOLD:g}: {int((density > DENSITY_HIGH_THRESHOLD).sum())} / {density.size}")
    print(f"|Vel| max = {np.sqrt(vx**2 + vy**2 + vz**2).max():.4f}")
    print(f"Cubos ativos: {cubos.sum()} / {cubos.size}")
    print(f"Celulas com warp pulado: {warp_info.sum()} / {warp_info.size}")
    if block_dims is not None:
        print(f"Block dim (threads): xBlockDim={block_dims[0]} yBlockDim={block_dims[1]} zBlockDim={block_dims[2]}")
    else:
        print("Aviso: cabecalho sem 'Block dim (threads): ...' — quadrados de bloco (amarelos) nao serao desenhados.")
    if total_time is not None:
        print(f"Tempo total de simulacao: {total_time:.6f} s")
    if gencubes_time is not None:
        print(f"Tempo de generateCubes: {gencubes_time:.6f} s")
    if total_threads_declared is not None:
        print(f"totalThreads (declarado no arquivo): {total_threads_declared}")
    if num_cubes is not None:
        print(
            f"Cubes Info: numCubes={num_cubes} "
            f"occupiedVolume={occupied_volume_pct:.2f}% "
            f"skippedWarps={skipped_warps_pct:.2f}%"
        )

    viewer = InteractiveSliceViewer(
        density, vx, vy, vz, cubos, warp_info, dims, cell_size, block_dims,
        total_time, gencubes_time,
        num_cubes, occupied_volume_pct, skipped_warps_pct,
    )
    viewer.show()

