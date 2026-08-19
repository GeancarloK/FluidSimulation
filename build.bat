@echo off
REM ============================================================
REM Script de compilacao do FluidSimulation via nvcc (linha de comando)
REM Ajuste as variaveis abaixo conforme seu ambiente antes de rodar
REM ============================================================
REM Arquitetura da GPU: RTX 3070 (Ampere) = sm_86
set ARCH=sm_86
REM Nome do executavel de saida
set OUTPUT=FluidSimulation.exe
REM Nivel de otimizacao (Release = -O3, Debug = -G -g)
set OPT_FLAGS=-O3
REM Caminho explicito do nvcc (v13.2 - v13.3 tem bug de crash no cudafe++
REM com o toolset do VS 2026, ver historico de troubleshooting)
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"

echo ============================================
echo Verificando se o nvcc (v13.2) esta disponivel...
echo ============================================
if not exist %NVCC% (
    echo.
    echo ERRO: nvcc nao encontrado em %NVCC%
    echo Confira se o CUDA Toolkit v13.2 esta instalado nesse caminho,
    echo ou ajuste a variavel NVCC neste script.
    echo.
    pause
    exit /b 1
)
%NVCC% --version
echo.
echo ============================================
echo Compilando %OUTPUT% para arquitetura %ARCH%...
echo ============================================
echo.
REM -allow-unsupported-compiler: necessario pois o Visual Studio instalado
REM (2026 / v18) e mais novo do que o CUDA 13.2 certifica oficialmente.
REM Normalmente ainda funciona, mas fique atento a erros estranhos de
REM compilacao que nao fariam sentido - podem vir dessa incompatibilidade.
%NVCC% -arch=%ARCH% %OPT_FLAGS% -std=c++17 -allow-unsupported-compiler main.cu kernels.cu utils.cu mesh.cu -o %OUTPUT%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================
    echo ERRO na compilacao. Veja as mensagens acima.
    echo ============================================
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo ============================================
echo Compilacao concluida com sucesso: %OUTPUT%
echo ============================================
dir %OUTPUT%
pause