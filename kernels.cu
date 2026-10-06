#include "hip/hip_runtime.h"
#include "defines.h"

#define TR_M 86095.961
/*
int T = 300;
double R = 8.314;
double M = 0.02897;
*/

__global__ void fluidMovement(
	const double* xVel0,
	const double* yVel0,
	const double* zVel0,
	const double* xArea,
	const double* yArea,
	const double* zArea,
	double* mass0,
	const char* warpInfo,
	int *progress,
	double deltaTime,
	double velFlux,
	double areaFlux,
	int xThreads,
	int yThreads,
	int zThreads,
	int xChunk,
	int yChunk,
	int zChunk,
	int gxBlocks,
	int gyBlocks,
	int sizeBlock)
{
	const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
	const int bdx = blockDim.x, bdy = blockDim.y, bdz = blockDim.z;

	// bloco GLOBAL = offset do chunk + posicao dentro do chunk
	const int bx = xChunk * (int)gridDim.x + (int)blockIdx.x;
	const int by = yChunk * (int)gridDim.y + (int)blockIdx.y;
	const int bz = zChunk * (int)gridDim.z + (int)blockIdx.z;

	const int x = tx + bdx * bx;
	const int y = ty + bdy * by;
	const int z = tz + bdz * bz;

	if (x >= xThreads || y >= yThreads || z >= zThreads) return;

	const int gdx = gxBlocks;
	const int gdy = gyBlocks;

	const int index = sizeBlock * (bx + gdx * (by + bz * gdy))
	                + (tx + bdx * (ty + tz * bdy));

	if (index == 0) atomicAdd_system(progress, 1);
	if (warpInfo[index]) return;

	const double xVel = (x == 0) ? velFlux  : xVel0[index];
	const double xA   = (x == 0) ? areaFlux : xArea[index];

	const double yVel = (y == 0) ? 0.0 : yVel0[index];
	const double yA   = (y == 0) ? 0.0 : yArea[index];

	const double zVel = (z == 0) ? 0.0 : zVel0[index];
	const double zA   = (z == 0) ? 0.0 : zArea[index];

	const bool noNextX = (x == xThreads - 1);
	const int xIndex_1B = noNextX ? index
	                    : (tx == bdx - 1 ? index + sizeBlock - tx
	                                     : index + 1);
	const double xVelN = noNextX ? velFlux  : xVel0[xIndex_1B];
	const double xAN   = noNextX ? areaFlux : xArea[xIndex_1B];

	const bool noNextY = (y == yThreads - 1);
	const int yIndex_1B = noNextY ? index
	                    : (ty == bdy - 1 ? index + sizeBlock * gdx - bdx * ty
	                                     : index + bdx);
	const double yVelN = noNextY ? 0.0 : yVel0[yIndex_1B];
	const double yAN   = noNextY ? 0.0 : yArea[yIndex_1B];

	const bool noNextZ = (z == zThreads - 1);
	const int zIndex_1B = noNextZ ? index
	                    : (tz == bdz - 1 ? index + sizeBlock * gdx * gdy
	                                       - bdx * bdy * tz
	                                     : index + bdx * bdy);
	const double zVelN = noNextZ ? 0.0 : zVel0[zIndex_1B];
	const double zAN   = noNextZ ? 0.0 : zArea[zIndex_1B];

	const double mass = mass0[index];

	double xVelEntry = xVel * xA;
	double xVelExit = xVelN * xAN;

	double yVelEntry = yVel * yA;
	double yVelExit = yVelN * yAN;

	double zVelEntry = zVel * zA;
	double zVelExit = zVelN * zAN;

	mass0[index] = mass + (xVelEntry - xVelExit + yVelEntry - yVelExit + zVelEntry - zVelExit) * deltaTime;
}

__global__ void recalculateVelocities(
	double* xVel0,
	double* yVel0,
	double* zVel0,
	const double* mass0,
	const double* xArea,
	const double* yArea,
	const double* zArea,
	const double* volume,
	const char* warpInfo,
	double beginMass,
	double deltaTime,
	double damping,
	int xThreads,
	int yThreads,
	int zThreads,
	int xChunk,
	int yChunk,
	int zChunk,
	int gxBlocks,
	int gyBlocks,
	int sizeBlock)
{
	const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
	const int bdx = blockDim.x, bdy = blockDim.y, bdz = blockDim.z;

	const int bx = xChunk * (int)gridDim.x + (int)blockIdx.x;
	const int by = yChunk * (int)gridDim.y + (int)blockIdx.y;
	const int bz = zChunk * (int)gridDim.z + (int)blockIdx.z;

	const int x = tx + bdx * bx;
	const int y = ty + bdy * by;
	const int z = tz + bdz * bz;

	if (x >= xThreads || y >= yThreads || z >= zThreads) return;

	const int gdx = gxBlocks;
	const int gdy = gyBlocks;

	const int index = sizeBlock * (bx + gdx * (by + bz * gdy))
	                + (tx + bdx * (ty + tz * bdy));

	if (warpInfo[index]) return;

	const double v = volume[index];
	const double m = mass0[index];

	const double xA = xArea[index];
	const int i_xm1 = (x == 0)  ? index
	                : (tx == 0) ? index - sizeBlock + bdx - 1
	                            : index - 1;
	const double m_xm1 = mass0[i_xm1];
	const double v_xm1 = volume[i_xm1];
	const double newVelX = xVel0[index];

	const double yA = yArea[index];
	const int i_ym1 = (y == 0)  ? index
	                : (ty == 0) ? index - sizeBlock * gdx + bdx * (bdy - 1)
	                            : index - bdx;
	const double m_ym1 = mass0[i_ym1];
	const double v_ym1 = volume[i_ym1];
	const double newVelY = yVel0[index];

	const double zA = zArea[index];
	const int i_zm1 = (z == 0)  ? index
	                : (tz == 0) ? index - sizeBlock * gdx * gdy
	                              + bdx * bdy * (bdz - 1)
	                            : index - bdx * bdy;
	const double m_zm1 = mass0[i_zm1];
	const double v_zm1 = volume[i_zm1];
	const double newVelZ = zVel0[index];

	const double rho = m * v;

	// X ---
	if (x != 0 && xA != 0.0)
	{
		const double deltaP = (m_xm1 * v_xm1 - rho) * TR_M;
		const double ax = deltaP * xA / (m + m_xm1);
		xVel0[index] = (newVelX + ax * deltaTime) * damping;
	}

	// Y ---
	if (y != 0 && yA != 0.0)
	{
		const double deltaP = (m_ym1 * v_ym1 - rho) * TR_M;
		const double ay = deltaP * yA / (m + m_ym1);
		yVel0[index] = (newVelY + ay * deltaTime) * damping;
	}

	// Z ---
	if (z != 0 && zA != 0.0)
	{
		const double deltaP = (m_zm1 * v_zm1 - rho) * TR_M;
		const double az = deltaP * zA / (m + m_zm1);
		zVel0[index] = (newVelZ + az * deltaTime) * damping;
	}
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

	float3 ray = RAY_DIR; // raio arbitrário já que o raio normal nao funcionou

	int frontHits = 0;
	int backHits = 0;


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

	const int sizeBlock = blockDim.x * blockDim.y * blockDim.z;
	const int index = sizeBlock * ((int)blockIdx.x + (int)gridDim.x
	                    * ((int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y))
	                + (threadIdx.x + blockDim.x * (threadIdx.y + threadIdx.z * blockDim.y));

	// se bateu o mesmo número de vezes de frente e de trás, está fora
	d_insideVertices[index] = (char)(frontHits < backHits);
}

__global__ void markWarpSkip(char* flags, int xThreads, int yThreads, int zThreads)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const int z = threadIdx.z + blockDim.z * blockIdx.z;

	const bool in = (x < xThreads) && (y < yThreads) && (z < zThreads);

	const int sizeBlock = blockDim.x * blockDim.y * blockDim.z;
	const int index = sizeBlock * ((int)blockIdx.x + (int)gridDim.x
	                    * ((int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y))
	                + (threadIdx.x + blockDim.x * (threadIdx.y + threadIdx.z * blockDim.y));

	const int i = in ? index : 0;

	const bool all = __all_sync(0xffffffff, in ? flags[i] != 0 : true);

	if (in) flags[i] = all;
}