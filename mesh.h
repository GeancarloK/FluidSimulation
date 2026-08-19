#pragma once
#ifndef MESH_H
#define MESH_H

#include "defines.h"


struct Triangle
{
    float3 centroid();
    float area();
    std::pair<float3, float3> dimensions();

    float3 a, b, c;
};

class Mesh
{
public:
    Mesh(const std::string& path);

    inline void add(Triangle tri) { triangles.push_back(tri); };
    inline void clear() { triangles.clear(); };

    std::pair<float3, float3> dimensions();
    float3 centroid();
    float3 size();
	void scale(float scale);
    void rotate90Z();
    void rotate270Z();

    void centerObjectToScene(float scale);
	std::vector<float> getVertices();

    std::vector<Triangle> triangles;
};

#endif //MESH_H