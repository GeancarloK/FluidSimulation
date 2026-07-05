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

void generateCubes(std::vector<bool>& cubos, std::vector<double>& mass, std::vector<double>& volume, std::vector<double>& areaX, std::vector<double>& areaY, std::vector<double>& areaZ)
{
	//float3 centroid = mesh.centroid();

	const float eighth = 1.0f / 8.0f;
	const float quarter = 1.0f / 4.0f;

	int xyThreads = xThreads * yThreads;

	int space = 7;

	for (int z = 1; z < zThreads; z++) //height
	{
		int z_yThreads = z * yThreads;

		for (int y = 1; y < yThreads; y++) //width
		{
			int y_z_yThreads_xThreads = (y + z_yThreads) * xThreads;

			for (int x = 1; x < xThreads; x++) //length
			{
				size_t indice = x + y_z_yThreads_xThreads;

				if (!(x% space > 3*space /4 && (y!= yThreads/2 || z != zThreads / 2))) continue; //skip
				cubos[indice] = true;

				//if(!cubos[indice]) continue;

				

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

void main()
{
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
	
	double *d_mass;
	double *d_volume;
	{
		cudaMalloc(&d_mass, totalThreads * sizeof(double));
		cudaMemcpy(d_mass, mass.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);

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

	double*xVel, *yVel, *zVel;
	{
		cudaMalloc(&xVel, totalThreads * sizeof(double));
		cudaMalloc(&yVel, totalThreads * sizeof(double));
		cudaMalloc(&zVel, totalThreads * sizeof(double));
		cudaMemcpy(xVel, lBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(yVel, wBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(zVel, hBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
	}




	double velFlux = VelFlux;
	double areaFlux = dyThreads * dzThreads;

	double deltaTime = 0.00001f;
	double totalTimeTeorical = 0.0f;
	double totalTimeReal = 0.0f;
	printf("blocksDim: %d %d %d\n", blocksDim.x, blocksDim.y, blocksDim.z);
	printf("threadsDim: %d %d %d\n", threadsDim.x, threadsDim.y, threadsDim.z);

	double damping = 0.9;
	double instDamping = pow(damping, deltaTime);

	double lastPrint = -1.0;
	int i = 0;
	//int maxIter = 300;
	double start = now();
	while (totalTimeTeorical < maxTime)
	{

		i++;
		
		fluidMovement<<<blocksDim, threadsDim>>>(
			xVel,
			yVel,
			zVel,
			d_xArea,
			d_yArea,
			d_zArea,
			d_mass,
			deltaTime,
			velFlux,
			areaFlux,	
			xThreads,
			yThreads,
			zThreads);
		//cudaError_t err = cudaGetLastError();
		//printf("Launch error: %s\n", cudaGetErrorString(err));
		checkCuda(cudaDeviceSynchronize(), "fluidMovement");

		recalculateVelocities<<<blocksDim, threadsDim>>>(
			xVel,
			yVel,
			zVel,
			d_mass,
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
		checkCuda(cudaDeviceSynchronize(), "recalculateVelocities");

		totalTimeTeorical += deltaTime;

		if (totalTimeTeorical>= 1.0)
		{
			totalTimeReal += now() - start;
			lastPrint = floor(totalTimeTeorical);

			// traz tudo do device de volta para o host
			cudaMemcpy(mass.data(), d_mass, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(volume.data(), d_volume, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(xArea.data(), d_xArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(yArea.data(), d_yArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(zArea.data(), d_zArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

			cudaMemcpy(lBorderVel.data(), xVel, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(wBorderVel.data(), yVel, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(hBorderVel.data(), zVel, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

			int xyThreads = xThreads * yThreads;

			FILE* dataFile = fopen("data.txt", "w");
			if (!dataFile) {
				fprintf(stderr, "Erro ao abrir data.txt para escrita\n");
				return;
			}

			fprintf(dataFile, "===== t=%.8lf s, iter=%d, velFlux=%.8lf =====\n", totalTimeTeorical, i, velFlux);
			fprintf(dataFile,
				"=== Grid Configuration ===\n"
				"Domain (m): length=%d  width=%d  height=%d\n"
				"numThreads=%d  numBlocks=%d\n\n"
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

			return; // limitando simulação a 1 segundo
			//system("pause");
			start = now();
		}
		
	}

	cudaMemcpy(mass.data(), d_mass, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
	for (double m : mass)
	{
		printf("mass: %lf\n", m);
	}
	printf("\n");
	printf("%d iterations, total time: %.8lf seconds\n", i, totalTimeTeorical);

	


	cudaFree(d_mass);
	cudaFree(d_volume);

	cudaFree(xVel);
	cudaFree(yVel);
	cudaFree(zVel);

	cudaFree(d_xArea);
	cudaFree(d_yArea);
	cudaFree(d_zArea);

	return;
}


