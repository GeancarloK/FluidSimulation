#include "utils.h"
#include "kernels.h"


int nxBlock; //numero de blocos
int nyBlock;
int nzBlock;

float dxBlock; //tamanho do bloco em metros
float dyBlock;
float dzBlock;

int nxThreads; //numero de threads por bloco
int nyThreads;
int nzThreads;

float dxThreads; //tamanho das threads em metros
float dyThreads;
float dzThreads;

int xThreads; //numero de threads totais
int yThreads;
int zThreads;

size_t totalThreads;

void generateCubes(std::vector<bool>& cubos, std::vector<double>& mass, std::vector<double>& volume, std::vector<double>& areaX, std::vector<double>& areaY, std::vector<double>& areaZ)
{
	const float eighth = 1.0f / 8.0f;
	const float quarter = 1.0f / 4.0f;

	int xyThreads = xThreads * yThreads;

	// paredes em 20%, 40%, 60%, 80% do comprimento
	const float wallCenters[] = {
		length * 0.2f,   // 10m
		length * 0.4f,   // 20m
		length * 0.6f,   // 30m
		length * 0.8f    // 40m
	};
	const float wallThick = 3.0f;  // espessura em metros
	const float wallMargin = 1.0f;  // distância das bordas do domínio
	const float holeHalf = 1.0f;  // metade do furo (2x2m → ±1m do centro)

	const float yCenterDomain = width / 2.0f;  // 10m
	const float zCenterDomain = height / 2.0f;  // 10m

	for (int z = 1; z < zThreads; z++)
	{
		float zPos = z * dzThreads;

		for (int y = 1; y < yThreads; y++)
		{
			float yPos = y * dyThreads;

			for (int x = 1; x < xThreads; x++)
			{
				float xPos = x * dxThreads;

				bool isWall = false;
				for (int w = 0; w < 4; w++)
				{
					float xMin = wallCenters[w] - wallThick / 2.0f;
					float xMax = wallCenters[w] + wallThick / 2.0f;

					if (xPos < xMin || xPos > xMax) continue;

					// margem: só existe parede entre 1m e 19m em y e z
					if (yPos < wallMargin || yPos > width - wallMargin) continue;
					if (zPos < wallMargin || zPos > height - wallMargin) continue;

					// furo 2x2m no centro da parede
					bool inHole = (yPos >= yCenterDomain - holeHalf &&
						yPos <= yCenterDomain + holeHalf &&
						zPos >= zCenterDomain - holeHalf &&
						zPos <= zCenterDomain + holeHalf);

					if (!inHole)
					{
						isWall = true;
						break;
					}
				}

				if (!isWall) continue;

				size_t indice = x + (y + z * yThreads) * xThreads;
				cubos[indice] = true;

				//float3 ponto = { x * dxThreads, y * dyThreads, z * dzThreads };
				volume[indice] -= eighth;
				volume[indice - 1] -= eighth;

				volume[indice - xThreads] -= eighth;
				volume[indice - 1 - xThreads] -= eighth;

				volume[indice - xyThreads] -= eighth;
				volume[indice - 1 - xyThreads] -= eighth;

				volume[indice - xThreads - xyThreads] -= eighth;
				volume[indice - 1 - xThreads - xyThreads] -= eighth;


				areaX[indice] -= quarter;
				areaX[indice - xThreads] -= quarter;
				areaX[indice - xyThreads] -= quarter;
				areaX[indice - xThreads - xyThreads] -= quarter;

				areaY[indice] -= quarter;
				areaY[indice - 1] -= quarter;
				areaY[indice - xyThreads] -= quarter;
				areaY[indice - 1 - xyThreads] -= quarter;

				areaZ[indice] -= quarter;
				areaZ[indice - 1] -= quarter;
				areaZ[indice - xThreads] -= quarter;
				areaZ[indice - 1 - xThreads] -= quarter;

			}
		}
	}

	for (size_t i = 0; i < totalThreads; i++)
	{
		mass[i] = volume[i];
	}
}

int run(size_t numBlocks, size_t numThreads, int iteration)
{
	totalThreads = numThreads* numBlocks;

	bestPartition(nxBlock, nyBlock, nzBlock, length, width, height, numBlocks);

	dxBlock = (float)length / nxBlock;
	dyBlock = (float)width / nyBlock;
	dzBlock = (float)height / nzBlock;
		

	dim3 blocksDim(nxBlock, nyBlock, nzBlock);

	bestPartition(nxThreads, nyThreads, nzThreads, dxBlock, dyBlock, dzBlock, numThreads);

	dxThreads = (float)dxBlock / nxThreads;
	dyThreads = (float)dyBlock / nyThreads;
	dzThreads = (float)dzBlock / nzThreads;

	xThreads = nxThreads * nxBlock;
	yThreads = nyThreads * nyBlock;
	zThreads = nzThreads * nzBlock;

	dim3 threadsDim(nxThreads, nyThreads, nzThreads);

	float volEsp = 0.8447f;
	double volThread = dxThreads * dyThreads * dzThreads;
	double beginMass = volThread / volEsp;

	std::vector<double> volume(totalThreads, 1);
	std::vector<double> mass(totalThreads, 1);


	std::vector<double> xArea(totalThreads, 1);
	std::vector<double> yArea(totalThreads, 1);
	std::vector<double> zArea(totalThreads, 1);

	std::vector<bool> cubos(totalThreads, false);

	generateCubes(cubos, mass, volume, xArea, yArea, zArea);
	
	
	for (double& m : mass)
	{
		m *= beginMass;
	}
	std::vector<int> iterThreads;
	iterThreads.reserve(totalThreads);
	for (size_t i = 0; i < totalThreads; i++)
	{
		double& v = volume[i];
		v *= volThread;
		if (volThread != 0.0f)
		{
			iterThreads.push_back(i);
		}
	}
	for (double& a : xArea)
	{
		a *= dyThreads * dzThreads;
	}
	for (double& a: yArea)
	{
		a *= dxThreads * dzThreads;
	}
	for (double& a : zArea)
	{
		a *= dxThreads * dyThreads;
	}
	
	double *d_mass0, *d_mass1;
	double *d_volume;
	{
		cudaMalloc(&d_mass0, totalThreads * sizeof(double));
		cudaMemcpy(d_mass0, mass.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc(&d_mass1, totalThreads * sizeof(double));
		cudaMemcpy(d_mass1, mass.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc(&d_volume, totalThreads * sizeof(double));
		cudaMemcpy(d_volume, volume.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
	}
	double* d_xArea, * d_yArea, * d_zArea;
	{
		cudaMalloc(&d_xArea, totalThreads * sizeof(double));
		cudaMalloc(&d_yArea, totalThreads * sizeof(double));
		cudaMalloc(&d_zArea, totalThreads * sizeof(double));
		cudaMemcpy(d_xArea, xArea.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_yArea, yArea.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_zArea, zArea.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
	}


	std::vector<double> lBorderVel(totalThreads, 0.0f);
	std::vector<double> wBorderVel(totalThreads, 0.0f);
	std::vector<double> hBorderVel(totalThreads, 0.0f);

	double *xVel0, *yVel0, *zVel0;
	double *xVel1, *yVel1, *zVel1;
	{
		cudaMalloc(&xVel0, totalThreads * sizeof(double));
		cudaMalloc(&yVel0, totalThreads * sizeof(double));
		cudaMalloc(&zVel0, totalThreads * sizeof(double));
		cudaMalloc(&xVel1, totalThreads * sizeof(double));
		cudaMalloc(&yVel1, totalThreads * sizeof(double));
		cudaMalloc(&zVel1, totalThreads * sizeof(double));
		cudaMemcpy(xVel0, lBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(yVel0, wBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(zVel0, hBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(xVel1, lBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(yVel1, wBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(zVel1, hBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
	}

	double velFlux = VelFlux;
	double areaFlux = dyThreads * dzThreads;

	double deltaTime = 0.00001f;
	double totalTimeTeorical = 0.0f;
	double totalTimeReal = 0.0f;
	//printf("blocksDim: %d %d %d\n", blocksDim.x, blocksDim.y, blocksDim.z);
	//printf("threadsDim: %d %d %d\n", threadsDim.x, threadsDim.y, threadsDim.z);

	double damping = 0.9;
	double instDamping = pow(damping, deltaTime);

	double lastPrint = -1.0;
	int i = 0;
	//int maxIter = 300;
	double start = now();
	while (totalTimeTeorical < maxTime)
	{

		i++;
		int iter = i % 2;
		if (iter == 0)
		{
			fluidMovement << <blocksDim, threadsDim >> > (
				xVel0,
				yVel0,
				zVel0,
				d_xArea,
				d_yArea,
				d_zArea,
				d_mass0,
				d_mass1,
				deltaTime,
				velFlux,
				areaFlux,
				xThreads,
				yThreads,
				zThreads);
			//cudaError_t err = cudaGetLastError();
			//printf("Launch error: %s\n", cudaGetErrorString(err));
			//checkCuda(cudaDeviceSynchronize(), "fluidMovement");

			recalculateVelocities << <blocksDim, threadsDim >> > (
				xVel0,
				yVel0,
				zVel0,
				xVel1,
				yVel1,
				zVel1,
				d_mass0,
				d_xArea,
				d_yArea,
				d_zArea,
				d_volume,
				beginMass,
				deltaTime,
				instDamping,
				xThreads,
				yThreads,
				zThreads);
			//err = cudaGetLastError();
			//printf("Launch error: %s\n", cudaGetErrorString(err));
			//checkCuda(cudaDeviceSynchronize(), "recalculateVelocities");
		}
		else
		{
			fluidMovement << <blocksDim, threadsDim >> > (
				xVel1,
				yVel1,
				zVel1,
				d_xArea,
				d_yArea,
				d_zArea,
				d_mass1,
				d_mass0,
				deltaTime,
				velFlux,
				areaFlux,
				xThreads,
				yThreads,
				zThreads);
			//cudaError_t err = cudaGetLastError();
			//printf("Launch error: %s\n", cudaGetErrorString(err));
			//checkCuda(cudaDeviceSynchronize(), "fluidMovement");

			recalculateVelocities << <blocksDim, threadsDim >> > (
				xVel1,
				yVel1,
				zVel1,
				xVel0,
				yVel0,
				zVel0,
				d_mass1,
				d_xArea,
				d_yArea,
				d_zArea,
				d_volume,
				beginMass,
				deltaTime,
				instDamping,
				xThreads,
				yThreads,
				zThreads);
			//err = cudaGetLastError();
			//printf("Launch error: %s\n", cudaGetErrorString(err));
			//checkCuda(cudaDeviceSynchronize(), "recalculateVelocities");
		}
		checkCuda(cudaDeviceSynchronize(), "iteraction");
		totalTimeTeorical += deltaTime;

		if (totalTimeTeorical>= 1.0)
		{
			totalTimeReal += now() - start;
			lastPrint = floor(totalTimeTeorical);

			// traz tudo do device de volta para o host
			cudaMemcpy(mass.data(), d_mass0, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(volume.data(), d_volume, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(xArea.data(), d_xArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(yArea.data(), d_yArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(zArea.data(), d_zArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

			cudaMemcpy(lBorderVel.data(), xVel0, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(wBorderVel.data(), yVel0, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(hBorderVel.data(), zVel0, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

			int xyThreads = xThreads * yThreads;

			char filename[256];
			snprintf(filename, sizeof(filename), "dataOpt_%zu_%zu.txt", numBlocks, numThreads);


			printf(
				"=== Grid Configuration ===\n"
				"Domain (m): length=%d  width=%d  height=%d\n"
				"numThreads=%zu  numBlocks=%zu\n\n"
				"Blocks: nxBlock=%d  nyBlock=%d  nzBlock=%d\n"
				"Block size (m): dxBlock=%.4f  dyBlock=%.4f  dzBlock=%.4f\n\n"
				"Threads per block: nxThreads=%d  nyThreads=%d  nzThreads=%d\n"
				"Thread size (m): dxThreads=%.4f  dyThreads=%.4f  dzThreads=%.4f\n\n"
				"Total threads: xThreads=%d  yThreads=%d  zThreads=%d\n"
				"Total simulation time (s): %.6f\n\n"
				"------------------------------------------------------------------\n\n",
				length, width, height,
				numThreads, numBlocks,
				nxBlock, nyBlock, nzBlock,
				dxBlock, dyBlock, dzBlock,
				nxThreads, nyThreads, nzThreads,
				dxThreads, dyThreads, dzThreads,
				xThreads, yThreads, zThreads,
				totalTimeReal);

			FILE* summaryFile = fopen("dataOpt.txt", "a");
			if (summaryFile)
			{
				fprintf(summaryFile,
					"=== Grid Configuration ===\n"
					"Domain (m): length=%d  width=%d  height=%d\n"
					"numThreads=%zu  numBlocks=%zu\n\n"
					"Blocks: nxBlock=%d  nyBlock=%d  nzBlock=%d\n"
					"Block size (m): dxBlock=%.4f  dyBlock=%.4f  dzBlock=%.4f\n\n"
					"Threads per block: nxThreads=%d  nyThreads=%d  nzThreads=%d\n"
					"Thread size (m): dxThreads=%.4f  dyThreads=%.4f  dzThreads=%.4f\n\n"
					"Total threads: xThreads=%d  yThreads=%d  zThreads=%d\n"
					"Total simulation time (s): %.6f\n\n"
					"------------------------------------------------------------------\n\n",
					length, width, height,
					numThreads, numBlocks,
					nxBlock, nyBlock, nzBlock,
					dxBlock, dyBlock, dzBlock,
					nxThreads, nyThreads, nzThreads,
					dxThreads, dyThreads, dzThreads,
					xThreads, yThreads, zThreads,
					totalTimeReal);
				fclose(summaryFile);
			}

			//
			if (iteration == 0)
			{
				FILE* dataFile = fopen(filename, "w");
				if (!dataFile) {
					fprintf(stderr, "Erro ao abrir %s para escrita\n", filename);
					break;
				}

				fprintf(dataFile, "===== t=%.8lf s, iter=%d, velFlux=%.8lf =====\n", totalTimeTeorical, i, velFlux);
				fprintf(dataFile,
					"=== Grid Configuration ===\n"
					"Domain (m): length=%d  width=%d  height=%d\n"
					"numThreads=%zu  numBlocks=%zu\n\n"
					"Blocks: nxBlock=%d  nyBlock=%d  nzBlock=%d\n"
					"Block size (m): dxBlock=%.4f  dyBlock=%.4f  dzBlock=%.4f\n\n"
					"Threads per block: nxThreads=%d  nyThreads=%d  nzThreads=%d\n"
					"Thread size (m): dxThreads=%.4f  dyThreads=%.4f  dzThreads=%.4f\n\n"
					"Total threads: xThreads=%d  yThreads=%d  zThreads=%d\n"
					"Total simulation time (s): %.6f\n",
					length, width, height,
					numThreads, numBlocks,
					nxBlock, nyBlock, nzBlock,
					dxBlock, dyBlock, dzBlock,
					nxThreads, nyThreads, nzThreads,
					dxThreads, dyThreads, dzThreads,
					xThreads, yThreads, zThreads,
					totalTimeReal);

				for (size_t k = 0; k < totalThreads; k++)
				{
					int z = k / xyThreads;
					int rem = k % xyThreads;
					int y = rem / xThreads;
					int x = rem % xThreads;

					double density = (volume[k] != 0.0) ? mass[k] / volume[k] : 0.0;

					fprintf(dataFile, "[%zu] (x=%d y=%d z=%d)  mass=%.4lf  volume=%.4f  density=%.4f  cubos=%d "
						"xArea=%.4f  yArea=%.4f  zArea=%.4f  "
						"xVel=%.4lf  yVel=%.4lf  zVel=%.4lf\n",
						k, x, y, z,
						mass[k], volume[k], density, (int)cubos[k],
						xArea[k], yArea[k], zArea[k],
						lBorderVel[k], wBorderVel[k], hBorderVel[k]);
				}
				fprintf(dataFile, "\n");
				fclose(dataFile);
			}
			//

			break;
			//system("pause");
			//start = now();
		}
		
	}
	


	cudaFree(d_mass0);
	cudaFree(d_mass1);
	cudaFree(d_volume);

	cudaFree(xVel0);
	cudaFree(yVel0);
	cudaFree(zVel0);
	cudaFree(xVel1);
	cudaFree(yVel1);
	cudaFree(zVel1);

	cudaFree(d_xArea);
	cudaFree(d_yArea);
	cudaFree(d_zArea);

	return 0;
}


int main()
{
	double start = now();
	// limpa arquivos de execuções anteriores
	remove("dataOpt.txt");

	std::vector<size_t> numBlocksList = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
	std::vector<size_t> numThreadsList = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

	for (size_t nb : numBlocksList)
		for (size_t nt : numThreadsList)
		{
			char filename[256];
			snprintf(filename, sizeof(filename), "dataOpt_%zu_%zu.txt", nb, nt);
			remove(filename);
		}

	int factorial = 5;
	for (int i = 1; i <= factorial; ++i)
	{
		for (size_t numBlocks : numBlocksList)
		{
			for (size_t numThreads : numThreadsList)
			{
				run(numBlocks, numThreads, i);
			}
		}
	}
	double totalTimeReal = now() - start;

	FILE* out = fopen("dataOpt.txt", "a");
	if (out)
	{
		fprintf(out, "\nRealTime: %f s\n", totalTimeReal);
		fclose(out);
	}

	return 0;
}