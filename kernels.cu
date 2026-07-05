#include "defines.h"

__global__ void fluidMovement(
	double* xVel,
	double* yVel,
	double* zVel,
	const double*  xArea,
	const double*  yArea,
	const double*  zArea,
	double*  mass,
	double deltaTime,
	double velFlux,
	double areaFlux,
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

	const int xIndex_1B = index + 1;
	const int yIndex_1B = index + xThreads;
	const int zIndex_1B = index + xyThreads;

	const double xVelEntry = (x == 0) ? velFlux * areaFlux : xVel[index] * xArea[index];
	const double xVelExit = (x == xThreads - 1) ? velFlux * areaFlux : xVel[xIndex_1B] * xArea[xIndex_1B];

	const double yVelEntry = (y == 0) ? 0.0 : yVel[index] * yArea[index];
	const double yVelExit = (y == yThreads - 1) ? 0.0 : yVel[yIndex_1B] * yArea[yIndex_1B];

	const double zVelEntry = (z == 0) ? 0.0 : zVel[index] * zArea[index];
	const double zVelExit = (z == zThreads - 1) ? 0.0 : zVel[zIndex_1B] * zArea[zIndex_1B];

	mass[index] += (xVelEntry - xVelExit + yVelEntry - yVelExit + zVelEntry - zVelExit) * deltaTime;
}

__global__ void recalculateVelocities(
	double* __restrict__ xVel,
	double* __restrict__ yVel,
	double* __restrict__ zVel,
	const double* __restrict__ mass,
	const double* __restrict__ xArea,
	const double* __restrict__ yArea,
	const double* __restrict__ zArea,
	const double* __restrict__ volume,
	double beginMass,
	double deltaTime,
	double damping,
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
	

	double v = __ldg(&volume[index]);

	if (v == 0) return;

	const double m = __ldg(&mass[index]);
	const double rho = m / v;

	/*
	int T = 300;
	double R = 8.314;
	double M = 0.02897;
	*/
	constexpr double TR_M = 86072.961; // T*R/M
	//const double damping = 0.99999; // 1 - deltaTime
	// X ---
	const double xA = __ldg(&xArea[index]);
	
	if (xA != 0 && x != 0)
	{
		const int i_xm1 = index - 1;
		const double m_xm1 = __ldg(&mass[i_xm1]);
		const double v_xm1 = __ldg(&volume[i_xm1]);
		const double deltaP = (m_xm1 / v_xm1 - rho) * TR_M;
		const double ax = deltaP * xA /  (m + m_xm1);
		xVel[index] += ax * deltaTime;
		xVel[index] *= damping; // damping
	}

	// Y ---
	const double yA = __ldg(&yArea[index]);
	
	if (yA != 0 && y != 0)
	{
		const int i_ym1 = index - xThreads;
		const double m_ym1 = __ldg(&mass[i_ym1]);
		const double v_ym1 = __ldg(&volume[i_ym1]);
		const double deltaP = (m_ym1 / v_ym1 - rho) * TR_M;
		const double ay = deltaP * yA / (m + m_ym1);
		yVel[index] += ay * deltaTime;
		yVel[index] *= damping; // damping	
	}

	// Z ---
	const double zA = __ldg(&zArea[index]);
	if (zA != 0 && z != 0)
	{
		const int i_zm1 = index - xyThreads;
		const double m_zm1 = __ldg(&mass[i_zm1]);
		const double v_zm1 = __ldg(&volume[i_zm1]);
		const double deltaP = (m_zm1 / v_zm1 - rho) * TR_M;
		const double az = deltaP * zA / (m + m_zm1);
		zVel[index] += az * deltaTime;
		zVel[index] *= damping; // damping
	}

}