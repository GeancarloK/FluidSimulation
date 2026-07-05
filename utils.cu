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

double now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

