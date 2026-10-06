#include "hip/hip_runtime.h"
#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "defines.h"

void bestPartition(int& nLength, int& nWidth, int& nHeight, float l, float w, float h, size_t N);

bool parseBool(const std::string& s);

hipDeviceProp_t getGpuProps();

void printGpuProperties();

void printHelp(const char* progName);

double now();

inline void checkCuda(hipError_t err, const char* msg)
{
    if (err != hipSuccess)
        printf("CUDA Error [%s]: %s\n", msg, hipGetErrorString(err));
};


#endif //UTILS_H