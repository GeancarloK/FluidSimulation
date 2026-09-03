#include "utils.h"
#include "mesh.h"
#include "kernels.h"

#define warpSize 32

#define damping 0.7
#define blocking 0.5f

float minTime = 0.01f;
float VelFlux = 12.0f/3.6f;
float maxTime = 1.0f;
float scale = 1.15f;
float deltaTime = 0.00001;
#define maxIter round(maxTime/ deltaTime)

bool freezeB = false;
bool freezeT = false;
bool write = false;

std::string object = "cargo.obj";
std::string folder = "data";

float length;
float width;
float height;

dim3 blocksDim;
dim3 threadsDim;

int nChunks = 1;

dim3 chunksDim;
dim3 chunkSize;

int nxBlock = 1; //numero de blocos
int nyBlock = 1;
int nzBlock = 1;

float dxBlock; //tamanho do bloco em metros
float dyBlock;
float dzBlock;

int nxThreads = 1; //numero de threads por bloco
int nyThreads = 1;
int nzThreads = 1;

float dxThreads; //tamanho das threads em metros
float dyThreads;
float dzThreads;

int xThreads; //numero de threads totais
int yThreads;
int zThreads;

size_t totalThreads;

std::pair<double, int> generateCubes(Mesh& object, std::vector<bool>& cubos, std::vector<double>& mass, std::vector<double>& volume, std::vector<double>& areaX, std::vector<double>& areaY, std::vector<double>& areaZ, double beginMass, double volThread)
{
	const float eighth = 1.0f / 8.0f;
	const float quarter = 1.0f / 4.0f;

	int cubes = 0;
	int xyThreads = xThreads * yThreads;

	float3 centerObject = object.centroid();

	std::vector<float> verticesObject = object.getVertices();
	float* d_verticesObject;
	cudaMalloc(&d_verticesObject, verticesObject.size() * sizeof(float));
	cudaMemcpy(d_verticesObject, verticesObject.data(), verticesObject.size() * sizeof(float), cudaMemcpyHostToDevice);

	std::vector<char> insideVertices(totalThreads, 0);
	char* d_insideVertices;
	cudaMalloc(&d_insideVertices, totalThreads * sizeof(char));
	cudaMemcpy(d_insideVertices, insideVertices.data(),totalThreads * sizeof(char), cudaMemcpyHostToDevice);

	double startObjectAnalysis = now();
	setInsideVertices << <blocksDim, threadsDim >> > (
		d_verticesObject,
		verticesObject.size() / 9,
		d_insideVertices, 
		centerObject.x,
		centerObject.y,
		centerObject.z,
		xThreads,
		yThreads,
		zThreads,
		dxThreads,
		dyThreads,
		dzThreads,
		length,
		width,
		height,
		1.0f/scale
		);
	checkCuda(cudaDeviceSynchronize(), "objectAnalysis");

	double elapsedInside = now() - startObjectAnalysis;
	printf("setInsideVertices: %.6f s\n", elapsedInside);

	cudaMemcpy(insideVertices.data(), d_insideVertices, totalThreads * sizeof(char), cudaMemcpyDeviceToHost);
	cudaFree(d_verticesObject);
	cudaFree(d_insideVertices);

	for (int z = 1; z < zThreads; z++)
	{
		float zPos = z * dzThreads;

		for (int y = 1; y < yThreads; y++)
		{
			float yPos = y * dyThreads;

			for (int x = 1; x < xThreads; x++)
			{
				float xPos = x * dxThreads;

				size_t indice = x + (y + z * yThreads) * xThreads;
				char isWall = insideVertices[indice];
				//printf("isWall[%zu] = %d\n", indice, (int)insideVertices[indice]);
				//continue;
				if (!isWall) continue;
				
				cubes++;
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
		mass[i] = beginMass * volume[i];
		volume[i] *= volThread;
	}
	return {elapsedInside, cubes};
}

int run(size_t numBlocks, size_t numThreads, std::string objPath)
{
	Mesh object(objPath);
	object.scale(1.0f / 20.0f);

	float3 size = object.size();

	length = size.x * scale;
	width = size.y * scale;
	height = size.z * scale;

	object.centerObjectToScene(scale);

	if(!freezeB)
	{
		if(freezeT) bestPartition(nxBlock, nyBlock, nzBlock, length/nxThreads, width/nyThreads, height/nzThreads, numBlocks);
		else bestPartition(nxBlock, nyBlock, nzBlock, length, width, height, numBlocks);
	}
	

	dxBlock = (float)length / nxBlock;
	dyBlock = (float)width / nyBlock;
	dzBlock = (float)height / nzBlock;


	blocksDim = dim3(nxBlock, nyBlock, nzBlock);

	if(!freezeT)
	{
		bestPartition(nxThreads, nyThreads, nzThreads, dxBlock, dyBlock, dzBlock, numThreads);
	}

	dxThreads = (float)dxBlock / nxThreads;
	dyThreads = (float)dyBlock / nyThreads;
	dzThreads = (float)dzBlock / nzThreads;

	xThreads = nxThreads * nxBlock;
	yThreads = nyThreads * nyBlock;
	zThreads = nzThreads * nzBlock;

	threadsDim = dim3(nxThreads, nyThreads, nzThreads);

	float volEsp = 0.8447f;
	double volThread = dxThreads * dyThreads * dzThreads;
	double beginMass = volThread / volEsp;

	std::vector<double> mass(totalThreads);
	std::vector<double> volume(totalThreads, 1);

	std::vector<double> xArea(totalThreads, 1);
	std::vector<double> yArea(totalThreads, 1);
	std::vector<double> zArea(totalThreads, 1);

	std::vector<bool> cubos(totalThreads, false);

	auto [generateCubesTime, numCubes] = generateCubes(object, cubos, mass, volume, xArea, yArea, zArea, beginMass, volThread);

	const double dyzThreads = dyThreads * dzThreads;
	const double dxzThreads = dxThreads * dzThreads;
	const double dxyThreads = dxThreads * dyThreads;

	for (double& a : xArea) a *= dyzThreads;
	for (double& a : yArea) a *= dxzThreads;
	for (double& a : zArea) a *= dxyThreads;

	double* d_mass;

		cudaMalloc(&d_mass, totalThreads * sizeof(double));
		cudaMemcpy(d_mass, mass.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);

	double* d_volume;
	cudaMalloc(&d_volume, totalThreads * sizeof(double));
	cudaMemcpy(d_volume, volume.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);

	double* d_xArea, * d_yArea, * d_zArea;
	cudaMalloc(&d_xArea, totalThreads * sizeof(double));
	cudaMalloc(&d_yArea, totalThreads * sizeof(double));
	cudaMalloc(&d_zArea, totalThreads * sizeof(double));
	cudaMemcpy(d_xArea, xArea.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yArea, yArea.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_zArea, zArea.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);

	std::vector<char> warpInfo(totalThreads);
	char* d_warpInfo;
	cudaMalloc(&d_warpInfo, totalThreads * sizeof(char));
	cudaMemset(d_warpInfo, 1, totalThreads * sizeof(char));

	std::vector<double> lBorderVel(totalThreads);
	std::vector<double> wBorderVel(totalThreads);
	std::vector<double> hBorderVel(totalThreads);

	for(int i = 0; i < totalThreads; i++)
	{
		lBorderVel[i] = xArea[i] > 0 ? VelFlux : 0.0;
	}

	double* xVel, * yVel, * zVel;

		cudaMalloc(&xVel, totalThreads * sizeof(double));
		cudaMalloc(&yVel, totalThreads * sizeof(double));
		cudaMalloc(&zVel, totalThreads * sizeof(double));

		cudaMemcpy(xVel, lBorderVel.data(), totalThreads * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemset(yVel, 0, totalThreads * sizeof(double));
		cudaMemset(zVel, 0, totalThreads * sizeof(double));




	int chunkX, chunkY, chunkZ;

	bestPartition(chunkX, chunkY, chunkZ, nxBlock, nyBlock, nzBlock, nChunks);
	chunksDim = dim3(chunkX, chunkY, chunkZ);
	chunkSize = dim3(nxBlock / chunkX, nyBlock / chunkY, nzBlock / chunkZ);

	assert(chunkSize.x > 0 && chunkSize.y > 0 && chunkSize.z > 0);

	auto CID = [&](int x, int y, int z) {
		return x + (y + z * chunksDim.y) * chunksDim.x;
	};

	cudaStream_t* streams = (cudaStream_t*)malloc(nChunks * sizeof(cudaStream_t));
	cudaEvent_t*  evMove  = (cudaEvent_t*) malloc(nChunks * sizeof(cudaEvent_t));
	cudaEvent_t*  evVel   = (cudaEvent_t*) malloc(nChunks * sizeof(cudaEvent_t));

	for (int i = 0; i < nChunks; ++i) {
		cudaStreamCreate(&streams[i]);
		cudaEventCreateWithFlags(&evMove[i], cudaEventDisableTiming);
		cudaEventCreateWithFlags(&evVel[i],  cudaEventDisableTiming);
	}


	int* h_progress;
	cudaHostAlloc(&h_progress, sizeof(int), cudaHostAllocMapped);
	*h_progress = 0;

	int* d_progress;
	cudaHostGetDevicePointer(&d_progress, h_progress, 0);
	volatile int* vProgress = (volatile int*)h_progress;

	//valores de entrada dos cubos do volume de controle
	double areaFlux = dyzThreads;

	double totalTimeTeorical = 0.0;
	double totalTimeReal = 0.0;

	//quantidade de energia é preservada por segundo
	
	double instDamping = pow(damping, deltaTime);

	int iter = 0;
	double start = now();
	int lastQueued = -1, lastRun = -1;

	while (iter <= maxIter)
	{
		for(int z = 0; z < chunksDim.z; z++)
		{
			for(int y = 0; y < chunksDim.y; y++)
			{
				for(int x = 0; x < chunksDim.x; x++)
				{
					const int c = CID(x, y, z);

					if (iter > 0)
					{
						if (x > 0)                    cudaStreamWaitEvent(streams[c], evVel[CID(x-1, y, z)], 0);
						if (x < (int)chunksDim.x - 1) cudaStreamWaitEvent(streams[c], evVel[CID(x+1, y, z)], 0);
						if (y > 0)                    cudaStreamWaitEvent(streams[c], evVel[CID(x, y-1, z)], 0);
						if (y < (int)chunksDim.y - 1) cudaStreamWaitEvent(streams[c], evVel[CID(x, y+1, z)], 0);
						if (z > 0)                    cudaStreamWaitEvent(streams[c], evVel[CID(x, y, z-1)], 0);
						if (z < (int)chunksDim.z - 1) cudaStreamWaitEvent(streams[c], evVel[CID(x, y, z+1)], 0);
					}

					fluidMovement<<<chunkSize, threadsDim, 0, streams[c]>>>(
						xVel,
						yVel,
						zVel,
						d_xArea,
						d_yArea,
						d_zArea,
						d_mass,
						d_volume,
						d_warpInfo,
						d_progress,
						deltaTime,
						VelFlux,
						areaFlux,
						xThreads,
						yThreads,
						zThreads,
						x,
						y,
						z
					);

					cudaEventRecord(evMove[c], streams[c]);
				}
			}
		}

		for(int z = 0; z < chunksDim.z; z++)
		{
			for(int y = 0; y < chunksDim.y; y++)
			{
				for(int x = 0; x < chunksDim.x; x++)
				{
					const int c = CID(x, y, z);

					if (x > 0)                    cudaStreamWaitEvent(streams[c], evMove[CID(x-1, y, z)], 0);
					if (x < (int)chunksDim.x - 1) cudaStreamWaitEvent(streams[c], evMove[CID(x+1, y, z)], 0);
					if (y > 0)                    cudaStreamWaitEvent(streams[c], evMove[CID(x, y-1, z)], 0);
					if (y < (int)chunksDim.y - 1) cudaStreamWaitEvent(streams[c], evMove[CID(x, y+1, z)], 0);
					if (z > 0)                    cudaStreamWaitEvent(streams[c], evMove[CID(x, y, z-1)], 0);
					if (z < (int)chunksDim.z - 1) cudaStreamWaitEvent(streams[c], evMove[CID(x, y, z+1)], 0);


					recalculateVelocities<<<chunkSize, threadsDim, 0, streams[c]>>> (
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
						blocking,
						xThreads,
						yThreads,
						zThreads,
						x,
						y,
						z
					);

					cudaEventRecord(evVel[c], streams[c]);
				}
			}
		}

		totalTimeTeorical += deltaTime;
		iter++;


		const int gpuIter    = *vProgress;
		const int pctQueued  = (int)(100.0 * iter / maxIter);
		const int pctRun     = (int)(100.0 * gpuIter / maxIter);

		if (pctQueued != lastQueued || pctRun != lastRun)
		{
			const double remain = (pctRun > 0) ? (100 - pctRun) * (now() - start) / pctRun : 0.0;
			printf("\rEnfileirado: %3d%%  |  Executado: %3d%% (%d/%d)  |  restante: %.1fs   ",
			       pctQueued, pctRun, gpuIter, (int)maxIter, remain);
			fflush(stdout);
			lastQueued = pctQueued;
			lastRun    = pctRun;
		}
	}

	const double deadline = now() + 3600.0;
	while (lastRun < 100 && now() < deadline)
	{
		const int gpuIter = *vProgress;
		const int pctRun  = (int)(100.0 * gpuIter / maxIter);

		if (pctRun != lastRun)
		{
			const double remain = (pctRun > 0) ? (100 - pctRun) * (now() - start) / pctRun : 0.0;
			printf("\rEnfileirado: 100%%  |  Executado: %3d%% (%d/%d)  |  restante: %.1fs   ",
			       pctRun, gpuIter, (int)maxIter, remain);
			fflush(stdout);
			lastRun = pctRun;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
	}

	printf("\n");
	checkCuda(cudaDeviceSynchronize(), "everything");
	totalTimeReal += now() - start;
	//lastPrint = floor(totalTimeTeorical);

	
	for (int i = 0; i < nChunks; ++i) {
		cudaStreamDestroy(streams[i]);
		cudaEventDestroy(evMove[i]);
		cudaEventDestroy(evVel[i]);
	}
	free(streams); free(evMove); free(evVel);

	// traz tudo do device de volta para o host
	cudaMemcpy(warpInfo.data(), d_warpInfo, totalThreads * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(mass.data(), d_mass, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

	int invalidSimulation = 0;

	if(write)
	{
		cudaMemcpy(volume.data(), d_volume, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(xArea.data(), d_xArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(yArea.data(), d_yArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(zArea.data(), d_zArea, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);

		cudaMemcpy(lBorderVel.data(), xVel, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(wBorderVel.data(), yVel, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(hBorderVel.data(), zVel, totalThreads * sizeof(double), cudaMemcpyDeviceToHost);
	}

	float skippedWarps = 0.0f;
	for(int c = 0; c < totalThreads; c++)
	{
		skippedWarps += warpInfo[c];
		invalidSimulation |= (mass[c] < 0);
	}
	skippedWarps *= 100.0f / totalThreads;

		printf(
		"=== Grid Configuration ===\n"
		"Domain (m): length=%.2f  width=%.2f  height=%.2f\n"
		"numThreads=%zu  numBlocks=%zu\n\n"
		"Blocks: nxBlock=%d  nyBlock=%d  nzBlock=%d\n"
		"Block size (m): dxBlock=%.4f  dyBlock=%.4f  dzBlock=%.4f\n\n"
		"Threads per block: nxThreads=%d  nyThreads=%d  nzThreads=%d\n"
		"Thread size (m): dxThreads=%.4f  dyThreads=%.4f  dzThreads=%.4f\n\n"
		"Total threads: xThreads=%d  yThreads=%d  zThreads=%d\n"
		"totalThreads=%d\n\n"
		"ValidSimulation=%d\n"
		"Cubes Info: numCubes=%d occupiedVolume=%.2f%% skippedWarps=%.2f%%\n"
		"generateCubes time (s): %.6f\n\n"
		"Total simulation time (s): %.6f\n\n"
		"------------------------------------------------------------------\n\n",
		length, width, height,
		numThreads, numBlocks,
		nxBlock, nyBlock, nzBlock,
		dxBlock, dyBlock, dzBlock,
		nxThreads, nyThreads, nzThreads,
		dxThreads, dyThreads, dzThreads,
		xThreads, yThreads, zThreads,
		(int)totalThreads,
		!invalidSimulation,
		numCubes, (double)numCubes * 100.0 / totalThreads, skippedWarps,
		generateCubesTime,
		totalTimeReal);

	system("mkdir -p data 2>/dev/null");


	char filename[256];
	snprintf(filename, sizeof(filename), "%s/dataOpt_%zu_%zu_%zu.txt",
		folder.c_str(), totalThreads, numBlocks, numThreads);

	FILE* dataFile;
	if(write) dataFile = fopen(filename, "w");
	else dataFile = fopen(filename, "a");

	if (dataFile)
	{
		fprintf(dataFile, "===== t=%.8lf s, iter=%d, velFlux=%.8lf =====\n", totalTimeTeorical, iter, VelFlux);
		fprintf(dataFile,
			"=== Grid Configuration ===\n"
			"Domain (m): length=%.2f  width=%.2f  height=%.2f\n"
			"numThreads=%zu  numBlocks=%zu\n\n"
			"Blocks: nxBlock=%d  nyBlock=%d  nzBlock=%d\n"
			"Block size (m): dxBlock=%.4f  dyBlock=%.4f  dzBlock=%.4f\n\n"
			"Threads per block: nxThreads=%d  nyThreads=%d  nzThreads=%d\n"
			"Thread size (m): dxThreads=%.4f  dyThreads=%.4f  dzThreads=%.4f\n\n"
			"Total threads: xThreads=%d  yThreads=%d  zThreads=%d\n"
			"totalThreads=%d\n\n"
			"ValidSimulation=%d\n"
			"Cubes Info: numCubes=%d occupiedVolume=%.2f%% skippedWarps=%.2f%%\n"
			"generateCubes time (s): %.6f\n\n"
			"Total simulation time (s): %.6f\n\n"
			"------------------------------------------------------------------\n\n",
			length, width, height,
			numThreads, numBlocks,
			nxBlock, nyBlock, nzBlock,
			dxBlock, dyBlock, dzBlock,
			nxThreads, nyThreads, nzThreads,
			dxThreads, dyThreads, dzThreads,
			xThreads, yThreads, zThreads,
			(int)totalThreads,
			!invalidSimulation,
			numCubes, (double)numCubes * 100.0 / totalThreads, skippedWarps,
			generateCubesTime,
			totalTimeReal);


		if(write)
		{
			// Para visualização das simulações
			int xyThreads = xThreads * yThreads;
			for (size_t k = 0; k < totalThreads; k++)
			{
				int z = k / xyThreads;
				int rem = k % xyThreads;
				int y = rem / xThreads;
				int x = rem % xThreads;

				double density = (volume[k] != 0.0) ? mass[k] / volume[k] : 0.0;

				fprintf(dataFile, "[%zu] (x=%d y=%d z=%d)  mass=%.4lf  volume=%.4f  density=%.4f  cubos=%d  warpskip=%d  "
					"xArea=%.4f  yArea=%.4f  zArea=%.4f  "
					"xVel=%.4lf  yVel=%.4lf  zVel=%.4lf\n",
					k, x, y, z,
					mass[k], volume[k], density, (int)cubos[k], (int)warpInfo[k],
					xArea[k], yArea[k], zArea[k],
					lBorderVel[k], wBorderVel[k], hBorderVel[k]);
			}
		}
		
		
		fprintf(dataFile, "\n");
		fclose(dataFile);
	}
	else
	{
		fprintf(stderr, "Erro ao abrir %s para escrita\n", filename);
	}

	cudaFreeHost(h_progress);

	cudaFree(d_warpInfo);
	cudaFree(d_volume);
	cudaFree(d_mass);
	cudaFree(xVel);
	cudaFree(yVel);
	cudaFree(zVel);

	cudaFree(d_xArea);
	cudaFree(d_yArea);
	cudaFree(d_zArea);

	return 0;
}

int main(int argc, char** argv)
{
	int numBlocks = 1;
	int numThreads = 1;
	bool recalc = false;

	for(int argi = 1; argi < argc; argi++)
	{
		std::string arg = argv[argi];

		if(arg == "--blocksDim")
		{
			nxBlock = std::stoi(argv[++argi]);
			nyBlock = std::stoi(argv[++argi]);
			nzBlock = std::stoi(argv[++argi]);
			numBlocks = nxBlock * nyBlock * nzBlock;
			recalc = false;
			freezeB = true;
		}
		else if(arg == "--threadsDim")
		{
			nxThreads = std::stoi(argv[++argi]);
			nyThreads = std::stoi(argv[++argi]);
			nzThreads = std::stoi(argv[++argi]);
			numThreads = nxThreads * nyThreads * nzThreads;
			freezeT = true;
		}
		else if(arg == "--numBlocks")
		{
			numBlocks =std::stoi(argv[++argi]);
			recalc = false;
			freezeB = false;
		}
		else if(arg == "--numThreads")
		{
			numThreads = std::stoi(argv[++argi]);
			freezeT = false;
		}
		else if(arg == "--problemSize")
		{
			totalThreads = std::stoi(argv[++argi]);
			recalc = true;
		}
		else if(arg == "--vel")
		{
			VelFlux = std::stof(argv[++argi]);
		}
		else if(arg == "--time")
		{
			maxTime = max(std::stof(argv[++argi]), minTime);
		}
		else if(arg == "--scale")
		{
			scale = std::stof(argv[++argi]);
		}
		else if(arg == "--deltaTime")
		{
			deltaTime = std::stof(argv[++argi]);
		}
		else if(arg == "--iter")
		{
			minTime = std::stof(argv[++argi]) * deltaTime;
			maxTime = minTime;
		}
		else if(arg == "--chunks")
		{
			nChunks = min(max(std::stoi(argv[++argi]), 1), numBlocks);
		}
		else if(arg == "--write")
		{
			write = parseBool(argv[++argi]);
		}
		else if(arg == "--object")
		{
			object = std::string(argv[++argi]) + ".obj";
		}
		else if(arg == "--folder")
		{
			folder = std::string(argv[++argi]);
		}
		else if(arg == "--deviceProperties")
		{
			printGpuProperties();
			return 0;
		}
		else if(arg == "--help")
		{
			printHelp(argv[0]);
			return 0;
		}
		else
		{
			printf("ERROR:%s\n", argv[argi]);
			return 1;
		}
	}

	if(recalc) numBlocks = totalThreads / numThreads;
	else totalThreads = numThreads * numBlocks;

	run(numBlocks, numThreads, object);

	return 0;
}