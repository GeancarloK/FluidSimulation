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

# ARGS e obrigatorio. O binario (main.cu) aceita flags nomeadas, em
# qualquer ordem e quantidade -- CHECK_ARGS so valida que algo foi
# passado; formato e validacao de valores sao responsabilidade do
# parser em main.cu.
CHECK_ARGS = $(if $(strip $(ARGS)),,\
    $(error ARGS nao informado. Exemplo: ARGS="--numThreads 1024 --numBlocks 64 --blocksDim 4 2 8"))

# Pasta onde o alvo "factorial" joga as saidas/dados do teste em lote.
DATA_DIR ?= data
DATA_FACT_DIR ?= data-factorial

# Listas testadas no fatorial -- todas as combinacoes de PROBLEM x THREADS
# serao executadas. Sobrescreva na linha de comando se quiser, ex:
#   make factorial PROBLEM="1000 5000" THREADS="128 256"
PROBLEM ?= 1048576 4194304 16777216
THREADS ?= 8 16 32 64 128 256 512 1024

XDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
YDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
ZDIV ?= 1 2 4 8 16 32 64

SRCS := main.cu kernels.cu utils.cu  mesh.cu
HDRS := defines.h kernels.h utils.h  mesh.h

ifeq ($(OS),Windows_NT)
    EXE       := .exe
    MKDIR      = if not exist "$(subst /,\,$(BUILD))" mkdir "$(subst /,\,$(BUILD))"
    RMDIR      = if exist "$(subst /,\,$(BUILD))" rmdir /S /Q "$(subst /,\,$(BUILD))"
    RMDIR_PATH = if exist "$(subst /,\,$1)" rmdir /S /Q "$(subst /,\,$1)"
    MKDIR_PATH = if not exist "$(subst /,\,$1)" mkdir "$(subst /,\,$1)"
    FIX        = $(subst /,\,$1)
    HOSTFLAGS :=
else
    EXE       :=
    MKDIR      = mkdir -p $(BUILD)
    RMDIR      = rm -rf $(BUILD)
    RMDIR_PATH = rm -rf $1
    MKDIR_PATH = mkdir -p $1
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

.PHONY: all run factorial thread-factorial clean help ncu ncu-setup ncu-quick ncu-full nsys 
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

# Numero de vezes que cada combinacao PROBLEM x THREADS e repetida no
# alvo "factorial". Deve ser um inteiro positivo.
#   make factorial REPEAT=5
REPEAT ?= 1
OBJECT ?= ball

CHECK_REPEAT = $(if $(shell test "$(REPEAT)" -gt 0 2>/dev/null && echo ok),,\
    $(error REPEAT invalido: "$(REPEAT)" -- precisa ser um inteiro positivo. Ex: REPEAT=5))

factorial: $(BIN)
	@$(CHECK_REPEAT)
	@$(call RMDIR_PATH,$(DATA_FACT_DIR))
	@$(call MKDIR_PATH,$(DATA_FACT_DIR))
	@echo Fatorial: PROBLEM={$(PROBLEM)} x THREADS={$(THREADS)} x REPEAT=$(REPEAT) = $(words $(PROBLEM)) x $(words $(THREADS)) x $(REPEAT) = $$(( $(words $(PROBLEM)) * $(words $(THREADS)) * $(REPEAT) )) execucoes
	@$(foreach r,$(shell seq 1 $(REPEAT)),$(foreach p,$(PROBLEM),$(foreach t,$(THREADS),echo -- rep=$(r) problemSize=$(p) numThreads=$(t) -- && $(call FIX,$(BIN)) --problemSize $(p) --numThreads $(t) --folder $(DATA_FACT_DIR) --write 0 --time 0 && )))echo Fatorial concluido.

CHECK_TF_ARGS = $(if $(strip $(NUMBLOCKS)),,\
    $(error NUMBLOCKS nao informado. Exemplo: make thread-factorial NUMBLOCKS=64 NUMTHREADS=1024 REPEAT=3))
CHECK_TF_ARGS += $(if $(strip $(NUMTHREADS)),,\
    $(error NUMTHREADS nao informado. Exemplo: make thread-factorial NUMBLOCKS=64 NUMTHREADS=1024 REPEAT=3))

# Varre todas as combinacoes XDIV x YDIV x ZDIV cujo produto x*y*z bate
# exatamente com NUMTHREADS, rodando:
#   BIN --numBlocks NUMBLOCKS --threadsDim x y z --write 0 --time 0
# repetido REPEAT vezes. Combinacoes cujo produto != NUMTHREADS sao puladas
# (o filtro roda em shell, ja que make nao compara aritmetica nativamente).
thread-factorial: $(BIN)
	@$(CHECK_TF_ARGS)
	@$(CHECK_REPEAT)
	@echo Thread-factorial: numBlocks=$(NUMBLOCKS) numThreads=$(NUMTHREADS) REPEAT=$(REPEAT) -- filtrando XDIV={$(XDIV)} x YDIV={$(YDIV)} x ZDIV={$(ZDIV)} por x*y*z=numThreads
	@for r in $(shell seq 1 $(REPEAT)); do \
	    for x in $(XDIV); do \
	        for y in $(YDIV); do \
	            for z in $(ZDIV); do \
	                if [ $$(( x * y * z )) -eq $(NUMTHREADS) ]; then \
	                    echo -- rep=$$r threadsDim=$$x $$y $$z -- ; \
	                    $(call FIX,$(BIN)) --numBlocks $(NUMBLOCKS) --threadsDim $$x $$y $$z --folder data-thread-factorial --write 0 --time 0 --object $(OBJECT) || exit 1; \
	                fi; \
	            done; \
	        done; \
	    done; \
	done; \
	echo Thread-factorial concluido.


# Perfila os kernels do LOOP (fluidMovement/recalculateVelocities), que são
# baratos por invocação — pode (e deve) rodar na escala real do problema
# (ex.: ARGS="--numBlocks 1024 --numThreads 1024"). Limitado a NCU_LAUNCHES
# invocações pra não gerar relatórios gigantes que travam o ncu-ui ao abrir.
ncu: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(BUILD)/ncu_report -f $(call FIX,$(BIN)) $(ARGS)

# Perfila SÓ o setInsideVertices, separado do loop. Ele roda 1x só, mas faz
# O(totalThreads x nTriangulos do .obj) trabalho, então --set full nele em
# escala real explode o tempo por passe do kernel replay e derruba o
# profiler (LaunchFailed). Use ARGS reduzido aqui (ex.: "--numBlocks 64
# --numThreads 64") — as métricas por-thread continuam representativas.
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
	@echo "make                                - compila (release, -O3)"
	@echo ""
	@echo "ARGS obrigatorio p/ run/ncu*/nsys, flags aceitas (qualquer ordem/quantidade):"
	@echo "  --blocksDim <x> <y> <z>           - fixa dimensoes exatas do grid de blocos (numBlocks = x*y*z)"
	@echo "  --threadsDim <x> <y> <z>          - fixa dimensoes exatas do bloco de threads (numThreads = x*y*z)"
	@echo "  --numBlocks <n>                   - total de blocos"
	@echo "  --numThreads <n>                  - total de threads por bloco"
	@echo "  --problemSize <n>                 - total de threads desejado; numBlocks e' recalculado (= n / numThreads)"
	@echo "  --vel <float>                     - velocidade do fluxo"
	@echo "  --time <float>                    - tempo maximo de simulacao (>= minTime)"
	@echo "  --scale <float>                   - fator de escala"
	@echo "  --deltaTime <float>               - passo de tempo"
	@echo "  --write <bool>                    - escreve saida (true/false ou 1/0)"
	@echo "  --object <nome>                   - nome do arquivo .obj (sem extensao)"
	@echo "  --deviceProperties                - mostra propriedades da GPU e sai"
	@echo "  -h, --help                        - mostra o help do binario e sai"
	@echo ""
	@echo "make run ARGS=\"--numBlocks 64 --numThreads 1024\""
	@echo "make ncu ARGS=\"--numBlocks 1024 --numThreads 1024\"       - profila fluidMovement/recalculateVelocities (escala real), limitado a NCU_LAUNCHES invocacoes"
	@echo "make ncu-setup ARGS=\"--numBlocks 64 --numThreads 64\"     - profila setInsideVertices ISOLADO; use escala reduzida (ele e' O(totalThreads x nTriangulos), full na escala real trava o profiler)"
	@echo "make ncu-quick ARGS=\"--numBlocks 64 --numThreads 1024\"   - profila todos os kernels com --set basic, rapido, para checagem inicial"
	@echo "make ncu-full ARGS=\"--numBlocks 64 --numThreads 1024\"    - profila TODAS as invocacoes do loop com --set full (relatorio pode ficar enorme)"
	@echo "make ncu ARGS=\"--numBlocks 1024 --numThreads 1024\" NCU_LAUNCHES=20 NCU_SKIP=100 - ajusta quantas invocacoes e a partir de qual pular"
	@echo "make nsys ARGS=\"--numBlocks 64 --numThreads 1024\"        - profila a execucao inteira com Nsight Systems"
	@echo "make factorial PROBLEM=\"100000 500000\" THREADS=\"128 256\" - limpa DATA_DIR e roda todas as combinacoes de PROBLEM x THREADS"
	@echo "make DEBUG=1                        - build com debug de device (-G -g)"
	@echo "make ARCH=sm_89                     - arch da GPU (sm_86, sm_89, native, ...)"
	@echo "make clean                          - remove a pasta build"