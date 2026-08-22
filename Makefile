NVCC   ?= nvcc
ARCH   ?= sm_86              # RTX 3070 Laptop (Ampere) = sm_86
STD    ?= c++17
TARGET ?= fluidsim
BUILD  ?= build
NCU         ?= ncu
NSYS        ?= nsys
NCU_KERNELS ?= fluidMovement|recalculateVelocities
NCU_SET     ?= full

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

.PHONY: all run clean help ncu nsys
.DEFAULT_GOAL := all

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(BUILD)/%.o: %.cu $(HDRS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD):
	$(MKDIR)

run: $(BIN)
	$(call FIX,$(BIN)) $(ARGS)
	
ncu: $(BIN)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" -o $(BUILD)/ncu_report -f $(call FIX,$(BIN)) $(ARGS)

nsys: $(BIN)
	$(NSYS) profile -o $(BUILD)/nsys_report -f true --trace=cuda,nvtx,osrt $(call FIX,$(BIN)) $(ARGS)

clean:
	$(RMDIR)

help:
	@echo "make            - compila (release, -O3)"
	@echo "make run        - compila e executa"
	@echo "make run ARGS=\"64 1024\" - compila e executa passando argv (numBlocks numThreads)"
	@echo "make ncu        - profila kernels (fluidMovement, recalculateVelocities) com Nsight Compute"
	@echo "make nsys       - profila a execucao inteira com Nsight Systems"
	@echo "make DEBUG=1    - build com debug de device (-G -g)"
	@echo "make ARCH=sm_89 - arch da GPU (sm_86, sm_89, native, ...)"
	@echo "make clean      - remove a pasta build"
	
