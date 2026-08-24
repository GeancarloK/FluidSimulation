NVCC   ?= nvcc
ARCH   ?= sm_86              # RTX 3070 Laptop (Ampere) = sm_86
STD    ?= c++17
TARGET ?= fluidsim
BUILD  ?= build
NCU         ?= ncu
NSYS        ?= nsys

# setInsideVertices roda 1x (setup, O(totalThreads x nTriangulos) - pesado)
# e é perfilado à parte, em escala reduzida (ver alvo ncu-setup).
# fluidMovement/recalculateVelocities rodam por timestep, são baratos e
# podem ser perfilados com --set full na escala real do problema.
NCU_LOOP_KERNELS  ?= fluidMovement|recalculateVelocities
NCU_SETUP_KERNELS ?= setInsideVertices
NCU_KERNELS       ?= $(NCU_LOOP_KERNELS)
NCU_SET     ?= full
NCU_LAUNCHES ?= 6            # nº de invocações de cada kernel a perfilar (0 = todas, cuidado!)
NCU_SKIP     ?= 0            # nº de invocações iniciais a pular (útil p/ ignorar warm-up)

# ARGS é obrigatório. O binário (main.cu) aceita exatamente estas formas,
# sempre com <numBlocks> e <numThreads> como os DOIS ÚLTIMOS argumentos:
#   <numBlocks> <numThreads>                    (2 args)
#   <time> <numBlocks> <numThreads>              (3 args)
#   <time> <speed> <numBlocks> <numThreads>      (4 args)
# CHECK_ARGS só valida a quantidade (2 a 4); a ordem é responsabilidade
# de quem chama. Usa só funções do Make (funciona igual no cmd.exe e no
# shell Unix).
MIN_ARGS := 2
MAX_ARGS := 4
CHECK_ARGS = $(if $(strip $(ARGS)),,\
    $(error ARGS nao informado. Exemplo: ARGS="--numThreads 1024 --numBlocks 64 --blocksDim 4 2 8"))

SRCS := main.cu kernels.cu utils.cu  mesh.cu
HDRS := defines.h kernels.h utils.h  mesh.h

ifeq ($(OS),Windows_NT)
    EXE       := .exe
    MKDIR      = if not exist "$(subst /,\,$(BUILD))" mkdir "$(subst /,\,$(BUILD))"
    RMDIR      = if exist "$(subst /,\,$(BUILD))" rmdir /S /Q "$(subst /,\,$(BUILD))"
    FIX        = $(subst /,\,$1)
    HOSTFLAGS :=
else
    EXE       :=
    MKDIR      = mkdir -p $(BUILD)
    RMDIR      = rm -rf $(BUILD)
    FIX        = $1
    HOSTFLAGS := -Xcompiler -fpermissive
endif

BIN  := $(BUILD)/$(TARGET)$(EXE)
OBJS := $(patsubst %.cu,$(BUILD)/%.o,$(SRCS))

ifeq ($(DEBUG),1)
    NVCCFLAGS ?= -G -g -std=$(STD) -arch=$(ARCH) $(HOSTFLAGS)
else
    NVCCFLAGS ?= -O3   -std=$(STD) -arch=$(ARCH) $(HOSTFLAGS)
endif

.PHONY: all run clean help ncu ncu-setup ncu-quick ncu-full nsys
.DEFAULT_GOAL := all

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(BUILD)/%.o: %.cu $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD):
	$(MKDIR)

run: $(BIN)
	@$(CHECK_ARGS)
	$(call FIX,$(BIN)) $(ARGS)

# Perfila os kernels do LOOP (fluidMovement/recalculateVelocities), que são
# baratos por invocação — pode (e deve) rodar na escala real do problema
# (ex.: ARGS="1024 1024"). Limitado a NCU_LAUNCHES invocações pra não gerar
# relatórios gigantes que travam o ncu-ui ao abrir.
ncu: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(BUILD)/ncu_report -f $(call FIX,$(BIN)) $(ARGS)

# Perfila SÓ o setInsideVertices, separado do loop. Ele roda 1x só, mas faz
# O(totalThreads x nTriangulos do .obj) trabalho, então --set full nele em
# escala real (ex. 1024 1024 -> ~1M threads) explode o tempo por passe do
# kernel replay e derruba o profiler (LaunchFailed). Use ARGS reduzido aqui
# (ex.: ARGS="64 64") — as métricas por-thread continuam representativas.
ncu-setup: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_SETUP_KERNELS)" \
	    -o $(BUILD)/ncu_report_setup -f $(call FIX,$(BIN)) $(ARGS)

# Sanidade rápida: set leve (basic), sem coleta pesada, útil para conferir
# se o binário/kernels estão sendo capturados antes de rodar o full.
# Inclui todos os kernels (setup + loop) pois o overhead do basic é baixo.
ncu-quick: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set basic -k "regex:$(NCU_SETUP_KERNELS)|$(NCU_LOOP_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(BUILD)/ncu_report_quick -f $(call FIX,$(BIN)) $(ARGS)

# Full "sem rede de proteção": perfila TODAS as invocações dos kernels do
# loop (NCU_KERNELS). Só use se souber que roda poucas vezes; senão o
# relatório fica enorme. Não inclui setInsideVertices (use ncu-setup).
ncu-full: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    -o $(BUILD)/ncu_report_full -f $(call FIX,$(BIN)) $(ARGS)

nsys: $(BIN)
	@$(CHECK_ARGS)
	$(NSYS) profile -o $(BUILD)/nsys_report -f true --trace=cuda,nvtx,osrt $(call FIX,$(BIN)) $(ARGS)

clean:
	$(RMDIR)

help:
	@echo "make                          - compila (release, -O3)"
	@echo ""
	help:
	@echo "make                          - compila (release, -O3)"
	@echo ""
	@echo "ARGS obrigatorio, flags aceitas (qualquer ordem/quantidade):"
	@echo "  --numBlocks <n>             - total de blocos (auto-particiona se --blocksDim nao usado)"
	@echo "  --numThreads <n>            - total de threads (auto-particiona se --threadsDim nao usado)"
	@echo "  --blocksDim <x> <y> <z>     - fixa dimensoes exatas do grid de blocos"
	@echo "  --threadsDim <x> <y> <z>    - fixa dimensoes exatas do bloco de threads"
	@echo "  --vel <float>               - velocidade do fluxo"
	@echo "  --time <float>              - tempo maximo de simulacao"
	@echo "  --deltaTime <float>         - passo de tempo"
	@echo ""
	@echo "make run ARGS=\"--numBlocks 64 --numThreads 1024\""
	@echo "make ncu ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\"
	@echo "make ncu ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\"     - profila fluidMovement/recalculateVelocities (escala real), limitado a NCU_LAUNCHES invocacoes"
	@echo "make ncu-setup ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\"  - profila setInsideVertices ISOLADO; use escala reduzida (ele e' O(totalThreads x nTriangulos), full na escala real trava o profiler)"
	@echo "make ncu-quick ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\" - profila todos os kernels com --set basic, rapido, para checagem inicial"
	@echo "make ncu-full ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\"  - profila TODAS as invocacoes do loop com --set full (relatorio pode ficar enorme)"
	@echo "make ncu ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\" NCU_LAUNCHES=20 NCU_SKIP=100 - ajusta quantas invocacoes e a partir de qual pular"
	@echo "make nsys ARGS=\"--blocksDim 16 8 8 --threadsDim 8 8 8\"      - profila a execucao inteira com Nsight Systems"
	@echo "make DEBUG=1                  - build com debug de device (-G -g)"
	@echo "make ARCH=sm_89               - arch da GPU (sm_86, sm_89, native, ...)"
	@echo "make clean                    - remove a pasta build"
