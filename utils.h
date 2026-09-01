#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "defines.h"

void bestPartition(int& nLength, int& nWidth, int& nHeight, float l, float w, float h, size_t N);

bool parseBool(const std::string& s);

cudaDeviceProp getGpuProps();

void printGpuProperties();

void printHelp(const char* progName);

double now();

inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
        printf("CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
};


#endif //UTILS_H