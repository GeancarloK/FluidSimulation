#pragma once
#ifndef KERNELS_H
#define KERNELS_H


__global__ void fluidMovement(
	double* xVel0,
	double* yVel0,
	double* zVel0,
	const double* xArea,
	const double* yArea,
	const double* zArea,
	double* mass0,
	double* mass1,
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
	double* __restrict__ xVel1,
	double* __restrict__ yVel1,
	double* __restrict__ zVel1,
	const double* __restrict__ mass0,
	const double* __restrict__ xArea,
	const double* __restrict__ yArea,
	const double* __restrict__ zArea,
	const double* __restrict__ volume,
	double beginMass,
	double deltaTime,
	double damping,
	int xThreads,
	int yThreads,
	int zThreads);
#endif