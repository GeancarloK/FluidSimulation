#include "defines.h"

__global__ void fluidMovement(
	const float* xVel0,
	const float* yVel0,
	const float* zVel0,
	const float* xArea,
	const float* yArea,
	const float* zArea,
	float* mass0,
	const float* volume,
	char* warpInfo,
	float deltaTime,
	float velFlux,
	float areaFlux,
	int xThreads,
	int yThreads,
	int zThreads)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int z = threadIdx.z + blockDim.z * blockIdx.z;

	if (x >= xThreads || y >= yThreads || z >= zThreads) return;

	const int xyThreads = xThreads * yThreads;
	const int index = x + y * xThreads + z * xyThreads;

	const bool empty = volume[index] == 0.0;

	const bool warpAllEmpty = __all_sync(__activemask(), empty);
	warpInfo[index] = warpAllEmpty;
	if(empty) return;

	const int xIndex_1B = index + 1;
	const int yIndex_1B = index + xThreads;
	const int zIndex_1B = index + xyThreads;

	const float xVelEntry = (x == 0) ? velFlux * areaFlux : xVel0[index] * xArea[index];
	const float xVelExit = (x == xThreads - 1) ? velFlux * areaFlux : xVel0[xIndex_1B] * xArea[xIndex_1B];

	const float yVelEntry = (y == 0) ? 0.0 : yVel0[index] * yArea[index];
	const float yVelExit = (y == yThreads - 1) ? 0.0 : yVel0[yIndex_1B] * yArea[yIndex_1B];

	const float zVelEntry = (z == 0) ? 0.0 : zVel0[index] * zArea[index];
	const float zVelExit = (z == zThreads - 1) ? 0.0 : zVel0[zIndex_1B] * zArea[zIndex_1B];

	mass0[index] += (xVelEntry - xVelExit + yVelEntry - yVelExit + zVelEntry - zVelExit) * deltaTime;
}

__global__ void recalculateVelocities(
	float* xVel0,
	float* yVel0,
	float* zVel0,
	const float* mass0,
	const float* xArea,
	const float* yArea,
	const float* zArea,
	const float* volume,
	float beginMass,
	float deltaTime,
	float damping,
	float blocking,
	int xThreads,
	int yThreads,
	int zThreads)
{

	int blockX = blockDim.x * blockIdx.x;
	int blockY = blockDim.y * blockIdx.y;
	int blockZ = blockDim.z * blockIdx.z;

	int x = threadIdx.x + blockX; // length
	int y = threadIdx.y + blockY; // width
	int z = threadIdx.z + blockZ; // height

	if (x >= xThreads || y >= yThreads || z >= zThreads) return;

	int xyThreads = xThreads * yThreads;

	int index = x + y * xThreads + z * xyThreads; // global index of the thread

	float v = volume[index];

	if (v == 0) return;
	float newVelX = xVel0[index];
	float newVelY = yVel0[index];
	float newVelZ = zVel0[index];

	const float m = mass0[index];
	//const double m = 5;
	const float rho = m / v;

	/*
	int T = 300;
	double R = 8.314;
	double M = 0.02897;
	*/
	constexpr float TR_M = 86095.961f; // T*R/M

	// X ---
	const float xA = xArea[index];

	if (xA != 0 && x != 0)
	{
		const int i_xm1 = index - 1;
		const float m_xm1 = mass0[i_xm1];
		const float v_xm1 = volume[i_xm1];
		const float deltaP = (m_xm1 / v_xm1 - rho) * TR_M;
		float ax = deltaP * xA / (m + m_xm1);
		newVelX = (newVelX + ax * deltaTime) * damping;
		if ((newVelX > 0 && m_xm1 <= 0) || (newVelX < 0 && m <= 0)) newVelX *= blocking;
	}


	// Y ---
	const float yA = yArea[index];

	if (yA != 0 && y != 0)
	{
		const int i_ym1 = index - xThreads;
		const float m_ym1 = mass0[i_ym1];
		const float v_ym1 = volume[i_ym1];
		const float deltaP = (m_ym1 / v_ym1 - rho) * TR_M;
		float ay = deltaP * yA / (m + m_ym1);
		newVelY = (newVelY + ay * deltaTime) * damping;
		if ((newVelY > 0 && m_ym1 <= 0) || (newVelY < 0 && m <= 0)) newVelY *= blocking;
	}


	// Z ---
	const float zA = zArea[index];

	if (zA != 0 && z != 0)
	{
		const int i_zm1 = index - xyThreads;
		const float m_zm1 = mass0[i_zm1];
		const float v_zm1 = volume[i_zm1];
		const float deltaP = (m_zm1 / v_zm1 - rho) * TR_M;
		float az = deltaP * zA / (m + m_zm1);
		newVelZ = (newVelZ + az * deltaTime) * damping;
		if ((newVelZ > 0 && m_zm1 <= 0) || (newVelZ < 0 && m <= 0)) newVelZ *= blocking;
	}

	xVel0[index] = newVelX;
	yVel0[index] = newVelY;
	zVel0[index] = newVelZ;
}







// -----------------------------------------------------
// Vector operations
// -----------------------------------------------------


__device__ float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}	

__device__ float3 cross(const float3& a, const float3& b)
{
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__device__ float3 minus(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


__constant__ float3 RAY_DIR = { 0.4082483f, 0.5345225f, 0.7407407f };

__global__ void setInsideVertices(
	const float* d_verticesObject,
	int numTriangles,
	char* d_insideVertices,
	float centerX,
	float centerY,
	float centerZ,
	int xThreads,
	int yThreads,
	int zThreads,
	float dxThreads,
	float dyThreads,
	float dzThreads,
	float length,
	float width,
	float height,
	float invScale
)
{

	int blockX = blockDim.x * blockIdx.x;
	int blockY = blockDim.y * blockIdx.y;
	int blockZ = blockDim.z * blockIdx.z;

	int x = threadIdx.x + blockX; // length
	int y = threadIdx.y + blockY; // width
	int z = threadIdx.z + blockZ; // height

	if (x >= xThreads || y >= yThreads || z >= zThreads) return;

	float3 pos = { x * dxThreads, y * dyThreads, z * dzThreads };

	/*
	float percX = pos.x / length;
	float percY = pos.y / width;
	float percZ = pos.z / height;

	float minCutoff = (1 - invScale) * 0.5f;
	float maxCutoff = 1 - minCutoff;

	if (percX < minCutoff || percX > maxCutoff || percY < minCutoff || percY > maxCutoff || percZ < minCutoff || percZ > maxCutoff) return;
	*/
	int xyThreads = xThreads * yThreads;

	float3 ray = RAY_DIR; // raio arbitrário já que o raio normal nao funcionou

	//float3 ray = { pos.x - centerX, pos.y - centerY , pos.z - centerZ};
	//float invLen = 1;//rsqrtf(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
	//ray.x *= invLen;
	//ray.y *= invLen;
	//ray.z *= invLen;
	

	int frontHits = 0;
	int backHits = 0;

	//__shared__ float3 verticesObject[3 * 1024];

	for (int t = 0; t < numTriangles; t++)
	{
		float3 a = { d_verticesObject[t * 9 + 0], d_verticesObject[t * 9 + 1], d_verticesObject[t * 9 + 2] };
		float3 b = { d_verticesObject[t * 9 + 3], d_verticesObject[t * 9 + 4], d_verticesObject[t * 9 + 5] };
		float3 c = { d_verticesObject[t * 9 + 6], d_verticesObject[t * 9 + 7], d_verticesObject[t * 9 + 8] };

		float3 ao = minus(a, pos);
		float3 ab = minus(b, a);
		float3 ac = minus(c, a);

		float3 crossAC = cross(ac, ray);

		float det = dot(ab, crossAC);          // sinal cru, antes de negar
		float invDet = -__frcp_rn(det);

		if (fabs(invDet) > 1e8f) continue;

		float beta = dot(ao, crossAC) * invDet;
		if (beta < 0.0f || beta > 1.0f) continue;

		float gamma = dot(ab, cross(ao, ray)) * invDet;
		if (gamma < 0.0f || beta + gamma > 1.0f) continue;

		float rayScale = -dot(ab, cross(ac, ao)) * invDet;
		if (rayScale <= 0.0f) continue;

		// acertou o triângulo com t>0: classifica de que lado o raio bateu
		if (det < 0.0f)
			frontHits++;
		else
			backHits++;
	}

	int index = x + y * xThreads + z * xyThreads;

	// se bateu o mesmo número de vezes de frente e de trás, está fora
	d_insideVertices[index] = (frontHits < backHits) ? 1 : 0;
}