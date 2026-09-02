"""
analyze_factorial.py

Analisa TODOS os arquivos gerados pelo alvo "factorial" do Makefile,
que ficam em data-factorial/dataOpt_<totalThreads>_<numBlocks>_<numThreads>.txt
-- um arquivo por combinacao de PROBLEM (totalThreads) x THREADS
(numThreads por bloco) testada, cada um com REPEAT execucoes anexadas.

Objetivo: comparar, para um MESMO totalThreads (primeiro numero do nome
do arquivo), como o tempo/eficiencia mudam conforme voce reparte esse
total em numBlocks x numThreads diferentes.

Uso:
    py analyze_factorial.py [pasta]

Se a pasta nao for informada, usa 'data-factorial' na mesma pasta deste
script. O script varre todos os arquivos "dataOpt_*.txt" dentro dela.

FORMATO DE ENTRADA / SIMULACOES INVALIDAS
-----------------------------------------
Cada execucao anexada ao dataFile traz o cabecalho "=== Grid
Configuration ===" com o campo `ValidSimulation=<0|1>` logo depois de
`totalThreads=`. Um arquivo (barra) e' considerado INVALIDO se qualquer
uma de suas REPEAT execucoes tiver ValidSimulation=0: a barra vira PRETA
no grafico por_total, ganha o sufixo "[invalido]" no rotulo e a coluna
`valid`/`nInvalid` no CSV. Arquivos no formato antigo, sem o campo, sao
tratados como validos.

Requer: numpy, matplotlib
    py -m pip install numpy matplotlib
"""
import os
import re
import sys
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "data-factorial")

# Ignora arquivos cujo numThreads (threads por bloco, o 3o numero do
# nome dataOpt_<total>_<numBlocks>_<numThreads>.txt) seja menor que
# este valor.
MIN_NUM_THREADS = 32

# Cor usada nas barras/rotulos cuja simulacao veio marcada como invalida
# (ValidSimulation=0).
INVALID_COLOR = "#000000"
# ======================================================================


FILENAME_RE = re.compile(r"dataOpt_(\d+)_(\d+)_(\d+)\.txt$")

# Marcador de inicio de registro: a linha "===== t=..." escrita pelo
# main.cu no comeco de cada execucao anexada (antes do cabecalho
# "=== Grid Configuration ===").
RECORD_START_RE = re.compile(r"===== t=")

# Numero em ponto flutuante tolerante a nan/inf/notacao cientifica.
NUM = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"

FIELD_RES = {
    "t":               re.compile(r"===== t=(" + NUM + r") s, iter=(\d+), velFlux=(" + NUM + r") ====="),
    "numThreads":      re.compile(r"numThreads=(\d+)"),
    "numBlocks":       re.compile(r"numBlocks=(\d+)"),
    "totalGrid":       re.compile(r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"),
    "totalThreads":    re.compile(r"totalThreads=(\d+)"),
    "validSimulation": re.compile(r"ValidSimulation=(-?\d+)"),
    "cubesInfo":       re.compile(r"Cubes Info:\s*numCubes=(\d+)\s+occupiedVolume=(" + NUM + r")%\s+skippedWarps=(" + NUM + r")%"),
    "gencubesTime":    re.compile(r"generateCubes time \(s\):\s*(" + NUM + r")"),
    "simTime":         re.compile(r"Total simulation time \(s\):\s*(" + NUM + r")"),
}


def find_data_files(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "dataOpt_*.txt")))
    files = []
    skipped_small_nt = 0
    for p in paths:
        m = FILENAME_RE.search(os.path.basename(p))
        if not m:
            print(f"Aviso: ignorando arquivo com nome inesperado: {p}", file=sys.stderr)
            continue
        total, numBlocks, numThreads = (int(v) for v in m.groups())
        if numThreads < MIN_NUM_THREADS:
            skipped_small_nt += 1
            continue
        files.append((total, numBlocks, numThreads, p))

    if skipped_small_nt:
        print(f"Ignorando {skipped_small_nt} arquivo(s) com numThreads < {MIN_NUM_THREADS}")

    return files


def split_records(text):
    """Quebra o texto em um registro por execucao usando '===== t=...'
    como marcador de inicio -- robusto independente de haver ou nao
    uma linha de tracos entre execucoes no dataFile."""
    starts = [m.start() for m in RECORD_START_RE.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def parse_record(record):
    m_sim_time = FIELD_RES["simTime"].search(record)
    if not m_sim_time:
        return None

    m_t = FIELD_RES["t"].search(record)
    m_nt = FIELD_RES["numThreads"].search(record)
    m_nb = FIELD_RES["numBlocks"].search(record)
    m_grid = FIELD_RES["totalGrid"].search(record)
    m_total = FIELD_RES["totalThreads"].search(record)
    m_cubes = FIELD_RES["cubesInfo"].search(record)
    m_gencubes = FIELD_RES["gencubesTime"].search(record)
    m_valid = FIELD_RES["validSimulation"].search(record)

    if m_grid:
        xThreads, yThreads, zThreads = (int(v) for v in m_grid.groups())
        total_cells = xThreads * yThreads * zThreads
    else:
        total_cells = None

    # ValidSimulation=0 -> simulacao invalida. Arquivos no formato
    # antigo (sem o campo) sao tratados como validos.
    validSimulation = (int(m_valid.group(1)) != 0) if m_valid else True

    return {
        "numThreads": int(m_nt.group(1)) if m_nt else None,
        "numBlocks": int(m_nb.group(1)) if m_nb else None,
        "totalCells": total_cells,
        "totalThreadsDeclared": int(m_total.group(1)) if m_total else None,
        "validSimulation": validSimulation,
        "hasValidField": m_valid is not None,
        "iter": int(m_t.group(2)) if m_t else None,
        "time": float(m_sim_time.group(1)),
        "gencubesTime": float(m_gencubes.group(1)) if m_gencubes else None,
        "numCubes": int(m_cubes.group(1)) if m_cubes else None,
        "occupiedVolumePct": float(m_cubes.group(2)) if m_cubes else None,
        "skippedWarpsPct": float(m_cubes.group(3)) if m_cubes else None,
    }


def parse_file(path):
    with open(path, "r") as f:
        text = f.read()
    entries = []
    for record in split_records(text):
        parsed = parse_record(record)
        if parsed is not None:
            entries.append(parsed)
    return entries


def build_stats(data_dir):
    """Le todos os arquivos da pasta e devolve:
    stats_by_total[totalThreads] = lista de dicts, um por arquivo
    (ou seja, por combinacao numBlocks/numThreads), com media/desvio
    das REPEAT execucoes daquele arquivo."""
    files = find_data_files(data_dir)
    if not files:
        print(f"Nenhum arquivo 'dataOpt_*.txt' encontrado em {data_dir} (apos filtro de numThreads >= {MIN_NUM_THREADS})", file=sys.stderr)
        sys.exit(1)

    stats_by_total = {}

    for total_from_name, numBlocks_from_name, numThreads_from_name, path in files:
        entries = parse_file(path)
        if not entries:
            print(f"Aviso: nenhuma execucao valida em {path}, pulando.", file=sys.stderr)
            continue

        # confere consistencia entre o nome do arquivo e o conteudo,
        # mas confia no NOME do arquivo (e' o que a Makefile usa pra
        # decidir qual arquivo e' qual)
        declared = entries[0]["totalThreadsDeclared"]
        if declared is not None and declared != total_from_name:
            print(
                f"Aviso: {os.path.basename(path)} diz totalThreads={total_from_name} "
                f"no nome mas o arquivo declara totalThreads={declared} -- usando o nome.",
                file=sys.stderr,
            )

        times = np.array([e["time"] for e in entries])
        effs = np.array([
            (e["totalCells"] * e["iter"] / e["time"])
            for e in entries
            if e["totalCells"] is not None and e["iter"] is not None and e["time"] > 0
        ])

        n_invalid = sum(1 for e in entries if not e["validSimulation"])

        stats_by_total.setdefault(total_from_name, []).append({
            "totalThreads": total_from_name,
            "numBlocks": numBlocks_from_name,
            "numThreads": numThreads_from_name,
            "n": len(entries),
            "valid": n_invalid == 0,
            "nInvalid": n_invalid,
            "meanTime": float(np.mean(times)),
            "stdTime": float(np.std(times)),
            "minTime": float(np.min(times)),
            "maxTime": float(np.max(times)),
            "meanEfficiency": float(np.mean(effs)) if effs.size else None,
            "stdEfficiency": float(np.std(effs)) if effs.size else None,
            "file": os.path.basename(path),
        })

    for total in stats_by_total:
        stats_by_total[total].sort(key=lambda s: s["numThreads"])

    return stats_by_total


def print_tables(stats_by_total):
    for total in sorted(stats_by_total):
        print(f"\n=== totalThreads = {total} ===")
        header = f"{'numBlocks':>10} {'numThreads':>10} {'n':>3} {'valid':>7}   {'tempo medio (s)':>16} {'desvio (s)':>12}   {'eficiencia media (cel/s)':>26}"
        print(header)
        print("-" * len(header))
        for s in stats_by_total[total]:
            eff_str = f"{s['meanEfficiency']:.3e}" if s["meanEfficiency"] is not None else "n/d"
            flag = "sim" if s["valid"] else f"NAO({s['nInvalid']})"
            print(
                f"{s['numBlocks']:>10} {s['numThreads']:>10} {s['n']:>3} {flag:>7}   "
                f"{s['meanTime']:>16.6f} {s['stdTime']:>12.6f}   "
                f"{eff_str:>26}"
            )


def save_csv(stats_by_total, outpath):
    rows = []
    for total in sorted(stats_by_total):
        rows.extend(stats_by_total[total])
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "totalThreads", "numBlocks", "numThreads", "n",
            "valid", "nInvalid",
            "meanTime", "stdTime", "minTime", "maxTime",
            "meanEfficiency", "stdEfficiency", "file",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  -> {outpath}")


def plot_overview(stats_by_total, outdir):
    """Uma linha por totalThreads, eixo x = numThreads (log2), mostrando
    como tempo/eficiencia mudam com a divisao numBlocks x numThreads --
    a mesma ideia do script anterior, mas agora 'total' e' o total real
    de threads do problema, nao apenas numBlocks."""
    totals = sorted(stats_by_total.keys())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(totals) - 1, 1)) for i in range(len(totals))]

    any_invalid = False
    for metric, ylabel, title, fname in [
        ("meanTime", "Tempo de simulacao (s)", "Tempo de simulacao vs numThreads por totalThreads", "overview_tempo.png"),
        ("meanEfficiency", "Eficiencia (cels/s)", "Eficiencia vs numThreads por totalThreads", "overview_eficiencia.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 7))
        for idx, total in enumerate(totals):
            rows = [s for s in stats_by_total[total] if s[metric] is not None]
            if not rows:
                continue
            nts = np.array([s["numThreads"] for s in rows])
            means = np.array([s[metric] for s in rows])
            stds = np.array([s["std" + metric[4:]] for s in rows])

            ax.plot(nts, means, "o-", color=colors[idx], label=f"total={total}", markersize=4, linewidth=1.2)
            ax.fill_between(nts, means - stds, means + stds, alpha=0.15, color=colors[idx])

            # Marca com um X preto os pontos cujo arquivo tem ao menos
            # uma execucao com ValidSimulation=0.
            inv_mask = np.array([not s["valid"] for s in rows])
            if inv_mask.any():
                any_invalid = True
                ax.scatter(nts[inv_mask], means[inv_mask], marker="x",
                           s=70, color="black", zorder=5, linewidths=1.6)

        if any_invalid:
            ax.scatter([], [], marker="x", s=70, color="black", linewidths=1.6,
                       label="ValidSimulation=0")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel("Threads per Block (numThreads)")
        ax.set_ylabel(ylabel + " — media ± desvio")
        ax.set_title(title)
        ax.legend(title="Total threads", fontsize=8, title_fontsize=9, ncol=2, loc="best")
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        outpath = os.path.join(outdir, fname)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"  -> {outpath}")


def plot_per_total(stats_by_total, outdir):
    """Um grafico de barras POR totalThreads, comparando as divisoes
    numBlocks x numThreads testadas para aquele total especifico --
    isso e' a comparacao 'mesmo total, splits diferentes' que voce
    pediu."""
    subdir = os.path.join(outdir, "por_total")
    os.makedirs(subdir, exist_ok=True)

    for total in sorted(stats_by_total):
        rows = sorted(stats_by_total[total], key=lambda s: s["numThreads"])
        labels = [
            f'B={s["numBlocks"]}\nT={s["numThreads"]}' + ("\n[invalido]" if not s["valid"] else "")
            for s in rows
        ]
        means = np.array([s["meanTime"] for s in rows])
        stds = np.array([s["stdTime"] for s in rows])

        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 5.5))
        x = np.arange(len(rows))
        palette = plt.cm.plasma(np.linspace(0, 1, len(rows)))
        colors = [c if s["valid"] else INVALID_COLOR for c, s in zip(palette, rows)]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.9,
               edgecolor="black", linewidth=0.4)

        # Nao ancora a base em 0: por padrao ax.bar forca ylim a incluir
        # 0, o que esmaga a diferenca entre barras quando os tempos ficam
        # todos numa faixa estreita e distante de zero. A base fica um
        # pouco abaixo do menor (media - desvio) e o topo um pouco acima
        # do maior (media + desvio), com folga de 5% do intervalo pra nao
        # cortar as barras de erro.
        lows = means - stds
        highs = means + stds
        span = highs.max() - lows.min()
        pad = span * 0.05 if span > 0 else max(highs.max() * 0.05, 1e-9)
        ax.set_ylim(lows.min() - pad, highs.max() + pad)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        for tick, s in zip(ax.get_xticklabels(), rows):
            if not s["valid"]:
                tick.set_color(INVALID_COLOR)
                tick.set_fontweight("bold")
        ax.set_ylabel("Tempo de simulacao (s) — media ± desvio")
        n_invalid = sum(1 for s in rows if not s["valid"])
        title = f"totalThreads = {total} — comparacao de splits numBlocks × numThreads"
        if n_invalid:
            title += f"\n{n_invalid} split(s) com ValidSimulation=0 em preto"
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        outpath = os.path.join(subdir, f"total_{total}.png")
        fig.savefig(outpath, dpi=200)
        plt.close(fig)

    print(f"  -> {subdir}/ ({len(stats_by_total)} graficos, um por totalThreads)")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR

    if not os.path.isdir(data_dir):
        print(f"Erro: pasta nao encontrada: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Lendo arquivos de: {data_dir}")
    stats_by_total = build_stats(data_dir)

    total_files = sum(len(v) for v in stats_by_total.values())
    print(f"{total_files} arquivos parseados, agrupados em {len(stats_by_total)} valores de totalThreads: "
          f"{sorted(stats_by_total.keys())}")

    n_invalid_files = sum(1 for rows in stats_by_total.values() for s in rows if not s["valid"])
    if n_invalid_files:
        print(f"  {n_invalid_files} arquivo(s) com ao menos uma execucao ValidSimulation=0 "
              f"(barras em preto no grafico por_total)")

    print_tables(stats_by_total)

    outdir = SCRIPT_DIR
    print("\nGerando saidas...")
    save_csv(stats_by_total, os.path.join(outdir, "factorial_resumo.csv"))
    plot_overview(stats_by_total, outdir)
    plot_per_total(stats_by_total, outdir)

    print("\nPronto.")


if __name__ == "__main__":
    main()