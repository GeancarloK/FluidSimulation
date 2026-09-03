#pragma once
#ifndef KERNELS_H
#define KERNELS_H


__global__ void fluidMovement(
	const double* __restrict__ xVel0,
	const double* __restrict__ yVel0,
	const double* __restrict__ zVel0,
	const double* __restrict__ xArea,
	const double* __restrict__ yArea,
	const double* __restrict__ zArea,
	double* __restrict__ mass0,
	const double* __restrict__ volume,
	char* __restrict__ warpInfo,
	double deltaTime,
	double velFlux,
	double areaFlux,
	int xThreads,
	int yThreads,
	int zThreads);

__global__ void recalculateVelocities(
	double* __restrict__ xVel0,
	double* __restrict__ yVel0,
	double* __restrict__ zVel0,
	const double* __restrict__ mass0,
	const double* __restrict__ xArea,
	const double* __restrict__ yArea,
	const double* __restrict__ zArea,
	const double* __restrict__ volume,
	double beginMass,
	double deltaTime,
	double damping,
	float blocking,
	int xThreads,
	int yThreads,
	int zThreads);

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
);

#endif