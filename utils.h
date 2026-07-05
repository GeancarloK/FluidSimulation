#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "defines.h"

void bestPartition(int& nLength, int& nWidth, int& nHeight, float l, float w, float h, size_t N);
double now();
inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
        printf("CUDA Error [%s]: %s\n", msg, cudaGetErrorString(err));
};


class Mesh
{
public: 
    struct triangle
    {
        float3 a, b, c;

        float3 centroidTri() const
        {
            float3 center = { 0.0f, 0.0f, 0.0f };
            center.x += a.x;
            center.y += a.y;
            center.z += a.z;

            center.x += b.x;
            center.y += b.y;
            center.z += b.z;

            center.x += c.x;
            center.y += c.y;
            center.z += c.z;

            center.x /= 3;
            center.y /= 3;
            center.z /= 3;

            return center;
        };

        float area() const
        {
            float3 ab = { b.x - a.x, b.y - a.y, b.z - a.z };
            float3 ac = { c.x - a.x, c.y - a.y, c.z - a.z };

            float3 cross = {
                ab.y * ac.z - ab.z * ac.y,
                ab.z * ac.x - ab.x * ac.z,
                ab.x * ac.y - ab.y * ac.x
            };

            return 0.5f * std::sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);
        };
    };

    std::vector<triangle> triangles;

    void add(triangle tri)
    {
        triangles.push_back(tri);
    };

    void clear()
    {
        triangles.clear();
    };

    float3 centroid() const
    {
        float3 center = { 0.0f, 0.0f, 0.0f };
        float totalArea = 0.0f;
        for (const triangle& tri : triangles)
        {
            const float a = tri.area();
            const float3 c = tri.centroidTri();
            center.x += c.x * a;
            center.y += c.y * a;
            center.z += c.z * a;
            totalArea += a;
        }
        center.x /= totalArea;
        center.y /= totalArea;
        center.z /= totalArea;
        return center;
    };
};

#endif