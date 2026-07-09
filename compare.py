"""
Compara dataOld.txt e dataOpt.txt, agrupando por (numThreads, numBlocks),
tira a media do "Total simulation time (s)" de cada grupo e gera um heatmap
(em %) da melhoria do Opt em relacao ao Old.

Uso:
    python compare_heatmap.py dataOld.txt dataOpt.txt
    (se nao passar argumentos, usa "dataOld.txt" e "dataOpt.txt" na pasta atual)

Requisitos: pandas, numpy, matplotlib
    pip install pandas numpy matplotlib
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Casa "numThreads=X  numBlocks=Y" seguido (em algum ponto depois) por
# "Total simulation time (s): Z", sem cruzar para o proximo bloco.
RECORD_PATTERN = re.compile(
    r"numThreads=(\d+)\s+numBlocks=(\d+).*?"
    r"Total simulation time \(s\):\s*([\d.]+)",
    re.DOTALL,
)


def parse_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    records = []
    for match in RECORD_PATTERN.finditer(text):
        num_threads = int(match.group(1))
        num_blocks = int(match.group(2))
        sim_time = float(match.group(3))
        records.append((num_threads, num_blocks, sim_time))

    if not records:
        raise ValueError(f"Nenhum registro encontrado em {path!r}. Verifique o formato do arquivo.")

    return pd.DataFrame(records, columns=["numThreads", "numBlocks", "time"])


def average_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por (numThreads, numBlocks) e tira a media do tempo de simulacao."""
    return (
        df.groupby(["numThreads", "numBlocks"], as_index=False)["time"]
        .mean()
        .rename(columns={"time": "time_mean"})
    )


def build_comparison(old_path: str, opt_path: str) -> pd.DataFrame:
    old_avg = average_by_config(parse_file(old_path)).rename(columns={"time_mean": "time_old"})
    opt_avg = average_by_config(parse_file(opt_path)).rename(columns={"time_mean": "time_opt"})

    merged = pd.merge(old_avg, opt_avg, on=["numThreads", "numBlocks"], how="inner")

    # Old / Opt em %: 100% = mesmo desempenho; >100% = Opt mais rapido que Old;
    # <100% = Opt mais lento que Old.
    merged["ratio_pct"] = merged["time_old"] / merged["time_opt"] * 100

    return merged


def plot_heatmap(merged: pd.DataFrame, output_path: str = "heatmap_opt_vs_old.png"):
    pivot = merged.pivot(index="numBlocks", columns="numThreads", values="ratio_pct")

    # ordena os eixos numericamente (numThreads/numBlocks costumam ser potencias de 2)
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    data = pivot.values

    # Escala linear e simetrica em torno de 100% (cor neutra), usando como raio
    # a distancia ate 250% (o lado que mais se afasta do centro). Isso faz o
    # vermelho em 90% aparecer bem discreto, ja que os dados tendem muito mais
    # pra cima de 100% do que pra baixo.
    VCENTER, VMAX_DISPLAY, VMIN_DISPLAY = 100, 240, 90
    half_range = VMAX_DISPLAY - VCENTER  # 150
    norm = Normalize(vmin=VCENTER - half_range, vmax=VMAX_DISPLAY)  # -50 a 250

    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.5 * len(pivot.index) + 3))
    # origin="lower" faz numBlocks crescer de baixo para cima no eixo Y
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=14)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=14)
    ax.set_xlabel("numThreads", fontsize=16)
    ax.set_ylabel("numBlocks", fontsize=16)
    ax.set_title(
        "Comparação de eficiência da otimização sobre a antiga implementação\n",
        fontsize=18,
    )

    # anota cada celula com o valor percentual
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=12, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Eficiência da otimização (%)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    # mostra a barra apenas a partir de 90% (o norm por baixo vai ate -50 so
    # para manter 100% centralizado, mas isso nao precisa aparecer)
    cbar.ax.set_ylim(VMIN_DISPLAY, VMAX_DISPLAY)
    cbar.set_ticks([90, 100, 130, 160, 190, 220, 240])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Heatmap salvo em: {output_path}")
    plt.close(fig)


def main():
    old_path = sys.argv[1] if len(sys.argv) > 1 else "dataOld.txt"
    opt_path = sys.argv[2] if len(sys.argv) > 2 else "dataOpt.txt"

    merged = build_comparison(old_path, opt_path)
    plot_heatmap(merged)


if __name__ == "__main__":
    main()