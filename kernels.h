#pragma once
#ifndef KERNELS_H
#define KERNELS_H


__global__ void fluidMovement(
	const float* __restrict__ xVel0,
	const float* __restrict__ yVel0,
	const float* __restrict__ zVel0,
	const float* __restrict__ xArea,
	const float* __restrict__ yArea,
	const float* __restrict__ zArea,
	float* __restrict__ mass0,
	const float* __restrict__ volume,
	char* __restrict__ warpInfo,
	float deltaTime,
	float velFlux,
	float areaFlux,
	int xThreads,
	int yThreads,
	int zThreads);

__global__ void recalculateVelocities(
	float* __restrict__ xVel0,
	float* __restrict__ yVel0,
	float* __restrict__ zVel0,
	const float* __restrict__ mass0,
	const float* __restrict__ xArea,
	const float* __restrict__ yArea,
	const float* __restrict__ zArea,
	const float* __restrict__ volume,
	float beginMass,
	float deltaTime,
	float damping,
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