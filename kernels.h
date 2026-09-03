#pragma once
#ifndef KERNELS_H
#define KERNELS_H


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
	int zThreads);

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