#include "mesh.h"


// -----------------------------------------------------------
// Triangle
// -----------------------------------------------------------


float3 Triangle::centroid()
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

float Triangle::area()
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

std::pair<float3, float3> Triangle::dimensions()
{
    float3 min = { a.x, a.y, a.z }, max = { a.x, a.y, a.z };

    min.x = b.x < min.x ? b.x : min.x;
    min.y = b.y < min.y ? b.y : min.y;
    min.z = b.z < min.z ? b.z : min.z;

    min.x = c.x < min.x ? c.x : min.x;
    min.y = c.y < min.y ? c.y : min.y;
    min.z = c.z < min.z ? c.z : min.z;

    max.x = b.x > max.x ? b.x : max.x;
    max.y = b.y > max.y ? b.y : max.y;
    max.z = b.z > max.z ? b.z : max.z;

    max.x = c.x > max.x ? c.x : max.x;
    max.y = c.y > max.y ? c.y : max.y;
    max.z = c.z > max.z ? c.z : max.z;

    return { min, max };
}

// -----------------------------------------------------------
// Mesh
// -----------------------------------------------------------

Mesh::Mesh(const std::string& path)
{
    this->clear();

    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);

    std::vector<float3> vertices;

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "v")
        {
            float3 v;
            iss >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        }
        else if (tag == "f")
        {
            std::vector<int> indices;
            std::string token;
            while (iss >> token)
            {
                // token pode ser "v", "v/vt", "v/vt/vn" ou "v//vn"
                size_t slashPos = token.find('/');
                std::string vStr = (slashPos == std::string::npos) ? token : token.substr(0, slashPos);

                int idx = std::stoi(vStr);
                if (idx < 0)
                    idx = static_cast<int>(vertices.size()) + idx + 1; // indice negativo (relativo ao final)

                indices.push_back(idx - 1); // OBJ e 1-indexado
            }

            // triangulacao em leque, caso a face tenha mais de 3 vertices
            for (size_t i = 1; i + 1 < indices.size(); ++i)
            {
                Triangle tri;
                tri.a = vertices[indices[0]];
                tri.b = vertices[indices[i]];
                tri.c = vertices[indices[i + 1]];
                this->add(tri);
            }
        }
    }
}

std::pair<float3, float3> Mesh::dimensions()
{
    float3 min = { FLT_MAX, FLT_MAX, FLT_MAX }, max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (Triangle& tri : triangles)
    {
        std::pair<float3, float3> triDim = tri.dimensions();
        min.x = triDim.first.x < min.x ? triDim.first.x : min.x;
        min.y = triDim.first.y < min.y ? triDim.first.y : min.y;
        min.z = triDim.first.z < min.z ? triDim.first.z : min.z;

        max.x = triDim.second.x > max.x ? triDim.second.x : max.x;
        max.y = triDim.second.y > max.y ? triDim.second.y : max.y;
        max.z = triDim.second.z > max.z ? triDim.second.z : max.z;
    }
    return { min, max };
}

float3 Mesh::centroid()
{
    std::pair<float3, float3> dim = dimensions();
    return { (dim.first.x + dim.second.x) / 2, (dim.first.y + dim.second.y) / 2, (dim.first.z + dim.second.z) / 2 };
};

float3 Mesh::size()
{
	std::pair<float3, float3> dim = dimensions();
	return { dim.second.x - dim.first.x, dim.second.y - dim.first.y, dim.second.z - dim.first.z };
};

void Mesh::centerObjectToScene(float scale)
{
    float3 dim = size();
	float3 centerObject = centroid();
    float3 centerScene = { scale * dim.x / 2.0f, scale * dim.y / 2.0f, scale * dim.z / 2.0f };

	float3 translation = { centerScene.x - centerObject.x, centerScene.y - centerObject.y, centerScene.z - centerObject.z };

	for (Triangle& tri : triangles)
	{
		tri.a.x += translation.x;
		tri.a.y += translation.y;
		tri.a.z += translation.z;

		tri.b.x += translation.x;
		tri.b.y += translation.y;
		tri.b.z += translation.z;

		tri.c.x += translation.x;
		tri.c.y += translation.y;
		tri.c.z += translation.z;
	}
}

void Mesh::scale(float scale)
{

	for (Triangle& tri : triangles)
	{
		tri.a.x *= scale;
		tri.a.y *= scale;
		tri.a.z *= scale;
		tri.b.x *= scale;
		tri.b.y *= scale;
		tri.b.z *= scale;
		tri.c.x *= scale;
		tri.c.y *= scale;
		tri.c.z *= scale;
	}
}

std::vector<float> Mesh::getVertices()
{
	std::vector<float> vertices;
	for (const Triangle& tri : triangles)
	{
		vertices.push_back(tri.a.x);
		vertices.push_back(tri.a.y);
		vertices.push_back(tri.a.z);
		vertices.push_back(tri.b.x);
		vertices.push_back(tri.b.y);
		vertices.push_back(tri.b.z);
		vertices.push_back(tri.c.x);
		vertices.push_back(tri.c.y);
		vertices.push_back(tri.c.z);
	}
	return vertices;
}

void Mesh::rotate90Z()
{
    for (Triangle& tri : triangles)
    {
        float ax = tri.a.x;
        tri.a.x = -tri.a.y;
        tri.a.y = ax;

        float bx = tri.b.x;
        tri.b.x = -tri.b.y;
        tri.b.y = bx;

        float cx = tri.c.x;
        tri.c.x = -tri.c.y;
        tri.c.y = cx;
    }
}


void Mesh::rotate270Z()
{
    for (Triangle& tri : triangles)
    {
        float ax = tri.a.x;
        tri.a.x = tri.a.y;
        tri.a.y = -ax;

        float bx = tri.b.x;
        tri.b.x = tri.b.y;
        tri.b.y = -bx;

        float cx = tri.c.x;
        tri.c.x = tri.c.y;
        tri.c.y = -cx;
    }
}