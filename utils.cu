#include "utils.h"

void bestPartition(int& nLength, int& nWidth, int& nHeight, float l, float w, float h, size_t N)
{
    float bestScore = FLT_MAX;

    for (int a = 1; a <= N; a++) {
        if (N % a != 0) continue;
        for (int b = 1; b <= N / a; b++) {
            if ((N / a) % b != 0) continue;
            int c = N / (a * b);

            float dx = (float)l / a, dy = (float)w / b, dz = (float)h / c;
            float lo = std::min({ dx, dy, dz });
            float hi = std::max({ dx, dy, dz });
            float score = hi / lo;

            if (score < bestScore) {
                bestScore = score;
                nLength = a; nWidth = b; nHeight = c;
            }
        }
    }
}

bool parseBool(const std::string& s)
{
    if (s == "1" || s == "true"  || s == "True"  || s == "TRUE")  return true;
    if (s == "0" || s == "false" || s == "False" || s == "FALSE") return false;
    throw std::runtime_error("valor booleano invalido: \"" + s + "\" (use true/false ou 1/0)");
}

cudaDeviceProp getGpuProps()
{
    int device;
    cudaGetDevice(&device); // pega o device atual (geralmente 0)

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop;
}

void printGpuProperties()
{
    cudaDeviceProp gpu = getGpuProps();

    int device;
    cudaGetDevice(&device);

    int clockRateKHz = 0, memClockRateKHz = 0;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device);
    cudaDeviceGetAttribute(&memClockRateKHz, cudaDevAttrMemoryClockRate, device);

    printf("=== Propriedades da GPU ===\n");
    printf("Nome: %s\n", gpu.name);
    printf("Compute capability: %d.%d\n", gpu.major, gpu.minor);
    printf("Multiprocessadores (SMs): %d\n", gpu.multiProcessorCount);
    printf("Clock: %.2f MHz\n", clockRateKHz / 1000.0);
    printf("\n");
    printf("Memoria global total: %.2f GB\n", gpu.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Memoria compartilhada por bloco: %zu KB\n", gpu.sharedMemPerBlock / 1024);
    printf("Memoria constante total: %zu KB\n", gpu.totalConstMem / 1024);
    printf("Cache L2: %d KB\n", gpu.l2CacheSize / 1024);
    printf("Clock memoria: %.2f MHz\n", memClockRateKHz / 1000.0);
    printf("Largura barramento memoria: %d bits\n", gpu.memoryBusWidth);
    printf("\n");
    printf("Warp size: %d\n", gpu.warpSize);
    printf("Registradores por bloco: %d\n", gpu.regsPerBlock);
    printf("Max threads por bloco: %d\n", gpu.maxThreadsPerBlock);
    printf("Max threadsDim: x=%d, y=%d, z=%d\n",
        gpu.maxThreadsDim[0], gpu.maxThreadsDim[1], gpu.maxThreadsDim[2]);
    printf("Max gridSize: x=%d, y=%d, z=%d\n",
        gpu.maxGridSize[0], gpu.maxGridSize[1], gpu.maxGridSize[2]);
}

void printHelp(const char* progName)
{
    printf("Uso: %s [opcoes]\n\n", progName);
    printf("Opcoes:\n");
    printf("  --blocksDim <x> <y> <z>    dimensoes do grid (numBlocks = x*y*z)\n");
    printf("  --threadsDim <x> <y> <z>   dimensoes do bloco (numThreads = x*y*z)\n");
    printf("  --numBlocks <n>            total de blocos\n");
    printf("  --numThreads <n>           threads por bloco\n");
    printf("  --problemSize <n>          total de threads desejado; numBlocks e' recalculado (= n / numThreads)\n");
    printf("  --vel <float>              velocidade do fluxo\n");
    printf("  --time <float>             tempo maximo de simulacao (>= minTime)\n");
    printf("  --scale <float>            fator de escala\n");
    printf("  --deltaTime <float>        passo de tempo\n");
    printf("  --write <bool>             escreve saida (true/false ou 1/0)\n");
    printf("  --object <nome>            nome do arquivo .obj (sem extensao)\n");
    printf("  --deviceProperties         mostra as propriedades da GPU e sai\n");
    printf("  -h, --help                 mostra esta mensagem e sai\n\n");
    printf("Exemplos:\n");
    printf("  %s --numBlocks 64 --numThreads 1024\n", progName);
    printf("  %s --blocksDim 16 8 8 --threadsDim 8 8 8\n", progName);
    printf("  %s --problemSize 1000000 --numThreads 256\n", progName);
}

double now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}





