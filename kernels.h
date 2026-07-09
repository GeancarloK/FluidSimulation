#pragma once
#ifndef KERNELS_H
#define KERNELS_H


__global__ void fluidMovement(
	double* xVel,
	double* yVel,
	double* zVel,
	const double* xArea,
	const double* yArea,
	const double* zArea,
	double* mass,
	double deltaTime,
	double velFlux,
	double areaFlux,
	int xThreads,
	int yThreads,
	int zThreads);

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
	int zThreads);
#endif