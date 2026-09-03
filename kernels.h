#pragma once
#ifndef KERNELS_H
#define KERNELS_H


__global__ void fluidMovement(
	const double* xVel0,
	const double* yVel0,
	const double* zVel0,
	const double* xArea,
	const double* yArea,
	const double* zArea,
	double* mass0,
	const double* volume,
	const char* warpInfo,
	double deltaTime,
	double velFlux,
	double areaFlux,
	int xThreads,
	int yThreads,
	int zThreads);

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