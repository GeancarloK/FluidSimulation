#!/usr/bin/env python3
"""Gerador de scripts .slurm para os experimentos do FluidSimulation no PCAD.

Estrutura assumida no cluster:

    ~/FluidSimulation/
        chunk/  early-memory/  float/  ...      <- uma pasta por experimento,
        main/   mul-add/  restrict/  final/        cada uma com seu Makefile
        experimentos/                            <- saidas (criada pelos jobs)
            main-tupi/  main-cidia/  ...
        jobs/                                    <- .slurm gerados por este script
        logs/                                    <- .out/.err dos jobs

Cada job (um por par experimento x maquina):
  1. recompila na propria maquina (arquitetura detectada da GPU do no; o binario
     fica rotulado por maquina, ex.: build/fluidsim-tupi-sm_89);
  2. roda ncu e nsys nas 3 configuracoes de interesse (curto e limitado);
  3. roda o thread-factorial completo (longo).

O profiling vem ANTES da varredura de proposito: ele e' curto e limitado por
--launch-count, enquanto a varredura pode consumir todo o wall clock. Nessa
ordem, um job que estoura o --time ainda entrega os relatorios.

Uso:
    python3 gerar_slurm.py                    # gera tudo com os defaults
    python3 gerar_slurm.py --experimentos main float --maquinas tupi
    python3 gerar_slurm.py --repeat 10 --totalthreads 4194304
    python3 gerar_slurm.py --gres            # adiciona --gres=gpu:1
"""

import argparse
import stat
from pathlib import Path

# ---------------------------------------------------------------- configuracao

EXPERIMENTOS = [
    "chunk",
    "early-memory",
    "float",
    "inv-volume",
    "main",
    "mul-add",
    "restrict",
    "final",
]

# tempo = limite de wall clock pedido ao Slurm. E' um TETO: o no e' liberado
# assim que o script termina, entao pedir mais nao "gasta" a maquina -- so
# pesa no escalonamento (jobs curtos costumam entrar antes).
# O PCAD permite ate 24h nessas particoes.
MAQUINAS = {
    "tupi":  {"particao": "tupi",  "tempo": "07:00:00"},   # 1x RTX 4090  (sm_89)
    "cidia": {"particao": "cidia", "tempo": "15:00:00"},   # 2x RTX 2080 Ti (sm_75)
}

# Configuracoes de threadsDim perfiladas com ncu/nsys em cada job.
DIMS_PROFILING = ["512 1 1", "16 16 1", "4 8 1"]

# Caminho ABSOLUTO: as diretivas #SBATCH nao expandem variaveis de shell
# ($HOME, ~), entao --output/--error precisam do caminho literal.
BASE_PADRAO = "/home/gkozenieski/FluidSimulation"
TOTALTHREADS_PADRAO = 1048576
REPEAT_PADRAO = 30

# ---------------------------------------------------------------- template

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={exp}-{maq}
#SBATCH --partition={particao}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={tempo}
#SBATCH --output={base}/logs/%x_%j.out
#SBATCH --error={base}/logs/%x_%j.err
{gres}
# Gerado por gerar_slurm.py -- nao edite a mao, edite o gerador.

set -u

BASE="{base}"
EXP="{exp}"
MAQ="{maq}"
SRC="$BASE/$EXP"
OUT="$BASE/experimentos/$EXP-$MAQ"

# O ambiente do job nao herda de forma confiavel o shell interativo, e o
# .bashrc pode nem ser lido: o CUDA e' exportado aqui explicitamente.
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

mkdir -p "$BASE/logs" "$OUT"

if [ ! -d "$SRC" ]; then
    echo "ERRO: pasta do experimento nao existe: $SRC"
    exit 1
fi
cd "$SRC" || exit 1

# --- proveniencia: sem isso, daqui a um mes ninguem sabe qual codigo gerou
# --- estes numeros, em qual GPU, com qual toolkit.
{{
    echo "job          : ${{SLURM_JOB_ID:-?}} (${{SLURM_JOB_NAME:-?}})"
    echo "no           : $(hostname)"
    echo "inicio       : $(date -Is)"
    echo "origem       : $SRC"
    echo "git branch   : $(git branch --show-current 2>/dev/null || echo n/a)"
    echo "git commit   : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "nvcc         : $(nvcc --version 2>/dev/null | tail -1 || echo ausente)"
    echo -n "gpu          : "
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null || echo n/a
    echo "binario      : $(make -s arch MACHINE="$MAQ" 2>/dev/null | sed -n 's/^BIN *= *//p')"
    echo "totalthreads : {tt}"
    echo "repeat       : {repeat}"
}} > "$OUT/execucao_${{SLURM_JOB_ID:-manual}}.info"

echo "=== build ==="
# MACHINE=$MAQ rotula o binario pela maquina do experimento (build/fluidsim-tupi-sm_89),
# em vez de depender do hostname do no sorteado pelo Slurm (tupi3, tupi5, ...).
make arch MACHINE="$MAQ"
if ! make MACHINE="$MAQ"; then
    echo "ERRO: build falhou em $SRC"
    exit 1
fi

echo
echo "=== profiling (ncu + nsys) ==="
# Primeiro porque e' curto e limitado; a varredura abaixo e' que arrisca
# bater no wall clock. Falha de profiling nao aborta o job.
for dim in {dims}; do
    echo "--- ncu  threadsDim=$dim ---"
    make ncu  MACHINE="$MAQ" TOTALTHREADS={tt} THREADSDIM="$dim" FOLDER="$OUT" \\
        || echo "AVISO: ncu falhou em '$dim' (segue o job)"
    echo "--- nsys threadsDim=$dim ---"
    make nsys MACHINE="$MAQ" TOTALTHREADS={tt} THREADSDIM="$dim" FOLDER="$OUT" \\
        || echo "AVISO: nsys falhou em '$dim' (segue o job)"
done

echo
echo "=== thread-factorial (varredura completa) ==="
# FORCE_REBUILD=1 (default do Makefile): recompila do zero antes de medir,
# garantindo que o binario da medicao e' o desta maquina e deste codigo.
make thread-factorial MACHINE="$MAQ" FORCE_REBUILD=1 \\
    TOTALTHREADS={tt} REPEAT={repeat} FOLDER="$OUT"
rc=$?

echo
echo "fim: $(date -Is)  (thread-factorial rc=$rc)"
echo "saidas em: $OUT"
exit $rc
"""

SUBMIT_TEMPLATE = """#!/bin/bash
# Submete todos os jobs gerados. Gerado por gerar_slurm.py.
cd "$(dirname "$0")" || exit 1

# O Slurm abre o arquivo de --output ANTES de rodar o script: se a pasta nao
# existir, o job falha sem deixar rastro. Por isso ela e' criada aqui.
mkdir -p "{base}/logs" "{base}/experimentos"

for f in {arquivos}; do
    echo -n "$f -> "
    sbatch "$f"
done
"""


def gerar(base, experimentos, maquinas, dims, tt, repeat, gres, saida):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    dims_bash = " ".join(f'"{d}"' for d in dims)
    gerados = []

    for exp in experimentos:
        for maq in maquinas:
            cfg = MAQUINAS[maq]
            texto = TEMPLATE.format(
                exp=exp,
                maq=maq,
                particao=cfg["particao"],
                tempo=cfg["tempo"],
                base=base,
                dims=dims_bash,
                tt=tt,
                repeat=repeat,
                gres="#SBATCH --gres=gpu:1\n" if gres else "",
            )
            caminho = saida / f"{exp}-{maq}.slurm"
            caminho.write_text(texto, encoding="utf-8")
            caminho.chmod(caminho.stat().st_mode | stat.S_IXUSR)
            gerados.append(caminho.name)

    submit = saida / "submeter_todos.sh"
    submit.write_text(
        SUBMIT_TEMPLATE.format(base=base, arquivos=" ".join(sorted(gerados))),
        encoding="utf-8",
    )
    submit.chmod(submit.stat().st_mode | stat.S_IXUSR)

    return gerados, submit


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=BASE_PADRAO,
                   help=f"raiz no cluster, ABSOLUTA (default: {BASE_PADRAO})")
    p.add_argument("--experimentos", nargs="+", default=EXPERIMENTOS,
                   help="pastas de experimento (default: todas as 8)")
    p.add_argument("--maquinas", nargs="+", default=list(MAQUINAS),
                   choices=list(MAQUINAS), help="maquinas alvo")
    p.add_argument("--dims", nargs="+", default=DIMS_PROFILING,
                   help='threadsDim perfilados, ex: --dims "512 1 1" "16 16 1"')
    p.add_argument("--totalthreads", type=int, default=TOTALTHREADS_PADRAO)
    p.add_argument("--repeat", type=int, default=REPEAT_PADRAO)
    p.add_argument("--gres", action="store_true",
                   help="inclui #SBATCH --gres=gpu:1")
    p.add_argument("--saida", default="jobs", help="pasta dos .slurm (default: jobs)")
    a = p.parse_args()

    gerados, submit = gerar(a.base, a.experimentos, a.maquinas, a.dims,
                            a.totalthreads, a.repeat, a.gres, a.saida)

    print(f"{len(gerados)} scripts gerados em {a.saida}/:")
    for g in sorted(gerados):
        print(f"  {g}")
    if not a.base.startswith("/"):
        print(f"\nAVISO: --base '{a.base}' nao e' absoluto. As diretivas #SBATCH "
              "nao expandem $HOME nem ~, e --output/--error vao falhar.")
    print(f"\nsubmeter todos: ./{submit}")
    print("submeter um:    sbatch jobs/main-tupi.slurm")


if __name__ == "__main__":
    main()