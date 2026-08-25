"""
generate_gif.py

Gera GIFs animados das fatias XY (variando z), XZ (variando y) e
YZ (variando x), com um unico vetor de velocidade RESULTANTE por
celula (fino, verde), partindo do CENTRO geometrico da celula.

O vetor central e' obtido promediando as velocidades de face da
grade staggered: para cada eixo do plano, a face de entrada e a
face de saida da celula sao somadas e divididas por 2 (equivalente
a "somar os 4 vetores" quando os dois eixos do plano sao
considerados juntos) — mesmo criterio usado no visualizer2Dpro.py.

Uso:
    python3 generate_gif.py <numBlocks> <numThreads>

O arquivo informado e' procurado dentro da pasta 'data', na mesma pasta
deste script (ex.: data/dataOpt_4194304_8192_512.txt).

Requer: numpy, matplotlib, pillow
    py -m pip install numpy matplotlib pillow
"""
import re
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from PIL import Image
 
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "frames")
 
# Tamanho fixo (em polegadas) e DPI usados em TODOS os frames. Isso e'
# essencial: se cada frame sair com um tamanho em pixels diferente (o que
# acontecia antes com bbox_inches="tight", que recorta a figura de forma
# variavel dependendo do texto do titulo/eixos), o GIF final acaba
# "vazando" pedacos de frames antigos por baixo dos novos (o formato GIF
# so redesenha a regiao coberta pelo frame novo; o resto do canvas
# permanece com o conteudo anterior). Isso e' exatamente o efeito de
# "aparecem outras imagens depois de antigos gifs" que voce estava vendo.
FIG_SIZE = (12, 8)
DPI = 100
# ======================================================================
 
HEADER_DIMS_RE = re.compile(
    r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"
)
HEADER_CELL_RE = re.compile(
    r"Thread size \(m\):\s*dxThreads=([\d.]+)\s+dyThreads=([\d.]+)\s+dzThreads=([\d.]+)"
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
ROW_RE = re.compile(
    r"\[(\d+)\]\s*\(x=(\d+)\s+y=(\d+)\s+z=(\d+)\)\s+"
    r"mass=([-\d.]+)\s+volume=([-\d.]+)\s+density=([-\d.]+)\s+"
    r"cubos=(\d+)\s*"
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
    gencubes_match = HEADER_GENCUBES_TIME_RE.search(text)
    total_threads_match = HEADER_TOTAL_THREADS_RE.search(text)
    cubes_info_match = HEADER_CUBES_INFO_RE.search(text)
    if not dims_match:
        raise ValueError("Nao encontrei 'Total threads: ...' no texto.")
    nx, ny, nz = (int(v) for v in dims_match.groups())
    dx, dy, dz = (float(v) for v in cell_match.groups()) if cell_match else (1.0, 1.0, 1.0)
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
 
    return {
        "density": density,
        "vel_x": vel_x,
        "vel_y": vel_y,
        "vel_z": vel_z,
        "cubos": cubos,
        "dims": (nx, ny, nz),
        "cell_size": (dx, dy, dz),
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
 
 
def render_slice(density_2d, cubos_2d, face_h_2d, face_v_2d,
                 h_size, v_size, nh, nv,
                 h_label, v_label, title, norm, filepath):
    """Renderiza uma fatia e salva como PNG.
 
    density_2d, cubos_2d: ja transpostos para [nv, nh] (mesma convencao
    do script original, para o imshow).
    face_h_2d, face_v_2d: NAO transpostos, formato [nh, nv] (mesma
    orientacao dos arrays vel_x/vel_y/vel_z fatiados), usados para
    calcular o vetor resultante no centro de cada celula.
    """
    # figsize/dpi fixos e SEM bbox_inches="tight": garante que todo PNG
    # gerado tenha exatamente o mesmo tamanho em pixels (FIG_SIZE[0]*DPI x
    # FIG_SIZE[1]*DPI). Usamos layout="constrained" para evitar que
    # titulos/labels sejam cortados, sem precisar recortar a figura.
    fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
 
    extent = [0, nh * h_size, 0, nv * v_size]
    ax.imshow(density_2d, origin="lower", extent=extent,
              cmap="coolwarm", norm=norm, aspect="equal",
              interpolation="nearest")
 
    # vetor resultante (fino, verde) partindo do centro de cada celula
    u_center, w_center = center_from_faces(face_h_2d, face_v_2d)
 
    centers_h = np.array([(i + 0.5) * h_size for i in range(nh)])
    centers_v = np.array([(j + 0.5) * v_size for j in range(nv)])
    C_h, C_v = np.meshgrid(centers_h, centers_v)   # [nv, nh]
 
    U = u_center.T   # [nv, nh]
    W = w_center.T   # [nv, nh]
 
    speed = np.sqrt(U**2 + W**2)
    max_speed = speed.max() if speed.max() > 0 else 1.0
    arrow_scale = max_speed * 15
 
    ax.quiver(C_h, C_v, U, W,
              color="green", alpha=0.85, scale=arrow_scale,
              width=0.0012, headwidth=2.5, headlength=3.5)
 
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
    # limites fixos (mesma escala/posicao em todo frame), reforcando o
    # tamanho identico do canvas entre frames
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
 
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
 
 
def make_gif(frame_dir, prefix, output_path, duration_ms=150):
    """Junta PNGs em GIF animado com qualidade consistente entre frames.
 
    Duas melhorias em relacao a versao anterior:
    1) Todos os frames sao forcados a ter o MESMO tamanho em pixels antes
       de virar GIF (defesa extra, alem do fix em render_slice) e cada
       frame usa disposal=2 (limpa o frame anterior antes de desenhar o
       proximo) — isso elimina o "vazamento"/ghosting de frames antigos.
    2) Em vez de deixar o Pillow quantizar cada PNG (que ja vem em RGB
       "full color") para 256 cores de forma INDEPENDENTE por frame (o
       que causa banding e flickering de cor entre frames — a principal
       causa da sensacao de "gif de baixa qualidade"), construimos uma
       unica paleta global a partir de uma amostra dos frames e
       quantizamos todos os frames para essa MESMA paleta.
    """
    files = sorted(
        [f for f in os.listdir(frame_dir) if f.startswith(prefix) and f.endswith(".png")],
        key=lambda f: int(re.search(r"(\d+)", f.replace(prefix, "")).group())
    )
    if not files:
        print(f"Nenhum frame encontrado com prefixo '{prefix}'")
        return
 
    frames = [Image.open(os.path.join(frame_dir, f)).convert("RGB") for f in files]
 
    # Garante tamanho identico em todos os frames (defesa extra)
    base_size = frames[0].size
    frames = [
        im if im.size == base_size else im.resize(base_size, Image.LANCZOS)
        for im in frames
    ]
 
    # Paleta global: monta uma "tira" com uma amostra dos frames e extrai
    # uma paleta de 256 cores a partir dela, para usar em TODOS os frames.
    sample_count = min(len(frames), 24)
    step = max(1, len(frames) // sample_count)
    sample_imgs = frames[::step]
    strip_w = sum(im.width for im in sample_imgs)
    strip = Image.new("RGB", (strip_w, base_size[1]))
    x = 0
    for im in sample_imgs:
        strip.paste(im, (x, 0))
        x += im.width
    palette_img = strip.quantize(colors=256, method=Image.MEDIANCUT)
 
    quantized_frames = [
        im.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG)
        for im in frames
    ]
 
    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"GIF salvo: {output_path} ({len(frames)} frames)")
 
 
def main():
    data_path = resolve_data_path(sys.argv)
    with open(data_path, "r") as f:
        text = f.read()
 
    parsed = parse_data(text)
    density = parsed["density"]
    vx, vy, vz = parsed["vel_x"], parsed["vel_y"], parsed["vel_z"]
    cubos = parsed["cubos"]
    nx, ny, nz = parsed["dims"]
    dx, dy, dz = parsed["cell_size"]
    gencubes_time = parsed["gencubes_time"]
    total_threads_declared = parsed["total_threads_declared"]
    num_cubes = parsed["num_cubes"]
    occupied_volume_pct = parsed["occupied_volume_pct"]
    skipped_warps_pct = parsed["skipped_warps_pct"]
 
    norm = Normalize(vmin=density.min(), vmax=density.max())
 
    print(f"Arquivo: {data_path}")
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
 
    # Limpa completamente a pasta de frames antes de gerar os novos.
    # Sem isso, PNGs de uma rodada anterior (possivelmente com outra
    # quantidade de fatias, ou dados de outro arquivo) ficam misturados
    # com os novos e acabam entrando no GIF final (make_gif so faz um
    # glob por prefixo, entao qualquer "xy_0031.png" antigo que sobrou
    # de uma rodada anterior e' incluido junto) — e' essa a causa mais
    # provavel de "aparecerem outras imagens depois" no final do GIF.
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # --- XY (variando z) ---
    print(f"Gerando {nz} fatias XY...")
    for iz in range(nz):
        render_slice(
            density_2d=density[:, :, iz].T,
            cubos_2d=cubos[:, :, iz].T,
            face_h_2d=vx[:, :, iz],
            face_v_2d=vy[:, :, iz],
            h_size=dx, v_size=dy, nh=nx, nv=ny,
            h_label="x (m)", v_label="y (m)",
            title=f"Plano XY — z={iz} (z_pos={iz*dz:.3f}m)",
            norm=norm,
            filepath=os.path.join(OUTPUT_DIR, f"xy_{iz:04d}.png")
        )
        print(f"  XY {iz+1}/{nz}", end="\r")
    print()
    make_gif(OUTPUT_DIR, "xy_", os.path.join(SCRIPT_DIR, "slices_xy.gif"), duration_ms=150)
 
    # --- XZ (variando y) ---
    print(f"Gerando {ny} fatias XZ...")
    for iy in range(ny):
        render_slice(
            density_2d=density[:, iy, :].T,
            cubos_2d=cubos[:, iy, :].T,
            face_h_2d=vx[:, iy, :],
            face_v_2d=vz[:, iy, :],
            h_size=dx, v_size=dz, nh=nx, nv=nz,
            h_label="x (m)", v_label="z (m)",
            title=f"Plano XZ — y={iy} (y_pos={iy*dy:.3f}m)",
            norm=norm,
            filepath=os.path.join(OUTPUT_DIR, f"xz_{iy:04d}.png")
        )
        print(f"  XZ {iy+1}/{ny}", end="\r")
    print()
    make_gif(OUTPUT_DIR, "xz_", os.path.join(SCRIPT_DIR, "slices_xz.gif"), duration_ms=150)
 
    # --- YZ (variando x) ---
    print(f"Gerando {nx} fatias YZ...")
    for ix in range(nx):
        render_slice(
            density_2d=density[ix, :, :].T,
            cubos_2d=cubos[ix, :, :].T,
            face_h_2d=vy[ix, :, :],
            face_v_2d=vz[ix, :, :],
            h_size=dy, v_size=dz, nh=ny, nv=nz,
            h_label="y (m)", v_label="z (m)",
            title=f"Plano YZ — x={ix} (x_pos={ix*dx:.3f}m)",
            norm=norm,
            filepath=os.path.join(OUTPUT_DIR, f"yz_{ix:04d}.png")
        )
        print(f"  YZ {ix+1}/{nx}", end="\r")
    print()
    make_gif(OUTPUT_DIR, "yz_", os.path.join(SCRIPT_DIR, "slices_yz.gif"), duration_ms=150)
 
    print("Pronto!")
 
 
if __name__ == "__main__":
    main()
 