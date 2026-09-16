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

std::pair<float, int> generateCubes(Mesh& objectMesh, std::vector<bool>& cubos, std::vector<float>& mass, std::vector<float>& volume, std::vector<float>& areaX, std::vector<float>& areaY, std::vector<float>& areaZ, float beginMass, float volThread)
{
	const float eighth = 1.0f / 8.0f;
	const float quarter = 1.0f / 4.0f;

	int cubes = 0;

	float elapsedInside = 0;

	std::vector<char> insideVertices(totalThreads, 0);

	// (x,y,z) -> indice linear por bloco. Mesma formula dos kernels:
	//   index = sizeBlock * (bx + gridDim.x*(by + bz*gridDim.y))
	//         + (tx + blockDim.x*(ty + tz*blockDim.y))
	auto idx = [](int x, int y, int z) -> size_t
	{
		const size_t sizeBlock = (size_t)nxThreads * nyThreads * nzThreads;

		const size_t bloco = (size_t)(x / nxThreads)
		                   + (size_t)nxBlock * ((size_t)(y / nyThreads)
		                   + (size_t)nyBlock * (size_t)(z / nzThreads));

		const size_t thread = (size_t)(x % nxThreads)
		                    + (size_t)nxThreads * ((size_t)(y % nyThreads)
		                    + (size_t)nyThreads * (size_t)(z % nzThreads));

		return sizeBlock * bloco + thread;
	};

	std::string nomeObjeto = object.substr(0, object.find_last_of('.'));

	// O threadsDim entra no nome porque o CONTEUDO depende dele: no layout
	// por bloco, a mesma grade produz arquivos diferentes para 8x8x4 e
	// 512x1x1. Sem isso a varredura leria o cache de uma configuracao dentro
	// de outra, com a geometria embaralhada e sem nenhum aviso.
	char filename[256];
	snprintf(filename, sizeof(filename), "cells/%s_%d_%d_%d_b%dx%dx%d.bin",
		nomeObjeto.c_str(), xThreads, yThreads, zThreads,
		nxThreads, nyThreads, nzThreads);

	std::filesystem::create_directories("cells");

	bool temCache = false;
	FILE* dataFile = fopen(filename, "rb");
	if (dataFile)
	{
		// Leitura curta nao e' aceita: um job morto no meio do fwrite deixa
		// um arquivo com o nome definitivo e metade do conteudo.
		const size_t lidos = fread(insideVertices.data(), sizeof(char), totalThreads, dataFile);
		fclose(dataFile);

		if (lidos == totalThreads)
		{
			temCache = true;
			printf("setInsideVertices: lido de %s\n", filename);
		}
		else
		{
			printf("AVISO: %s tem %zu celulas, esperadas %zu; recalculando.\n",
				filename, lidos, (size_t)totalThreads);
			std::fill(insideVertices.begin(), insideVertices.end(), 0);
		}
	}

	if (!temCache)
	{
		float3 centerObject = objectMesh.centroid();

		std::vector<float> verticesObject = objectMesh.getVertices();
		float* d_verticesObject;
		cudaMalloc(&d_verticesObject, verticesObject.size() * sizeof(float));
		cudaMemcpy(d_verticesObject, verticesObject.data(), verticesObject.size() * sizeof(float), cudaMemcpyHostToDevice);

		char* d_insideVertices;
		cudaMalloc(&d_insideVertices, totalThreads * sizeof(char));
		cudaMemcpy(d_insideVertices, insideVertices.data(), totalThreads * sizeof(char), cudaMemcpyHostToDevice);

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

		elapsedInside = now() - startObjectAnalysis;
		printf("setInsideVertices: %.6f s\n", elapsedInside);

		cudaMemcpy(insideVertices.data(), d_insideVertices, totalThreads * sizeof(char), cudaMemcpyDeviceToHost);
		cudaFree(d_verticesObject);
		cudaFree(d_insideVertices);

		// Grava num temporario e so' entao renomeia: o rename e' atomico,
		// entao ou o arquivo final existe inteiro, ou nao existe.
		char tmpname[300];
		snprintf(tmpname, sizeof(tmpname), "%s.tmp", filename);

		FILE* out = fopen(tmpname, "wb");
		if (!out)
		{
			printf("AVISO: nao foi possivel escrever %s; seguindo sem cache.\n", tmpname);
		}
		else
		{
			const size_t escritos = fwrite(insideVertices.data(), sizeof(char), totalThreads, out);
			fclose(out);

			if (escritos != totalThreads)
			{
				printf("AVISO: escrita de %s incompleta; descartando.\n", tmpname);
				remove(tmpname);
			}
			else if (rename(tmpname, filename) != 0)
			{
				printf("AVISO: falha ao renomear %s -> %s.\n", tmpname, filename);
				remove(tmpname);
			}
			else
			{
				printf("cache gravado: %s\n", filename);
			}
		}
	}

	for (int z = 1; z < zThreads; z++)
	{
		float zPos = z * dzThreads;

		for (int y = 1; y < yThreads; y++)
		{
			float yPos = y * dyThreads;

			for (int x = 1; x < xThreads; x++)
			{
				float xPos = x * dxThreads;

				// Os oito cantos do cubo. No layout por bloco os vizinhos NAO
				// sao deslocamentos constantes (-1, -xThreads, -xyThreads):
				// cada canto vem das suas proprias coordenadas.
				// Nomenclatura: i<dx><dy><dz>, 1 = deslocado de -1 no eixo.
				const size_t i000 = idx(x,     y,     z    );
				const size_t i100 = idx(x - 1, y,     z    );
				const size_t i010 = idx(x,     y - 1, z    );
				const size_t i110 = idx(x - 1, y - 1, z    );
				const size_t i001 = idx(x,     y,     z - 1);
				const size_t i101 = idx(x - 1, y,     z - 1);
				const size_t i011 = idx(x,     y - 1, z - 1);
				const size_t i111 = idx(x - 1, y - 1, z - 1);

				char isWall = insideVertices[i000];
				if (!isWall) continue;

				cubes++;
				cubos[i000] = true;

				volume[i000] -= eighth;
				volume[i100] -= eighth;

				volume[i010] -= eighth;
				volume[i110] -= eighth;

				volume[i001] -= eighth;
				volume[i101] -= eighth;

				volume[i011] -= eighth;
				volume[i111] -= eighth;


				areaX[i000] -= quarter;
				areaX[i010] -= quarter;
				areaX[i001] -= quarter;
				areaX[i011] -= quarter;

				areaY[i000] -= quarter;
				areaY[i100] -= quarter;
				areaY[i001] -= quarter;
				areaY[i101] -= quarter;

				areaZ[i000] -= quarter;
				areaZ[i100] -= quarter;
				areaZ[i010] -= quarter;
				areaZ[i110] -= quarter;

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
	const bool interativo = (std::getenv("SLURM_JOB_ID") == nullptr);

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
	float volThread = dxThreads * dyThreads * dzThreads;
	float beginMass = volThread / volEsp;

	std::vector<float> mass(totalThreads);
	std::vector<float> volume(totalThreads, 1);

	std::vector<float> xArea(totalThreads, 1);
	std::vector<float> yArea(totalThreads, 1);
	std::vector<float> zArea(totalThreads, 1);

	std::vector<bool> cubos(totalThreads, false);

	auto [generateCubesTime, numCubes] = generateCubes(object, cubos, mass, volume, xArea, yArea, zArea, beginMass, volThread);

	const float dyzThreads = dyThreads * dzThreads;
	const float dxzThreads = dxThreads * dzThreads;
	const float dxyThreads = dxThreads * dyThreads;

	for (float& a : xArea) a *= dyzThreads;
	for (float& a : yArea) a *= dxzThreads;
	for (float& a : zArea) a *= dxyThreads;

	float* d_mass;

		cudaMalloc(&d_mass, totalThreads * sizeof(float));
		cudaMemcpy(d_mass, mass.data(), totalThreads * sizeof(float), cudaMemcpyHostToDevice);

	float* d_volume;
	cudaMalloc(&d_volume, totalThreads * sizeof(float));
	cudaMemcpy(d_volume, volume.data(), totalThreads * sizeof(float), cudaMemcpyHostToDevice);

	float* d_xArea, * d_yArea, * d_zArea;
	cudaMalloc(&d_xArea, totalThreads * sizeof(float));
	cudaMalloc(&d_yArea, totalThreads * sizeof(float));
	cudaMalloc(&d_zArea, totalThreads * sizeof(float));
	cudaMemcpy(d_xArea, xArea.data(), totalThreads * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yArea, yArea.data(), totalThreads * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_zArea, zArea.data(), totalThreads * sizeof(float), cudaMemcpyHostToDevice);

	std::vector<char> warpInfo(totalThreads);
	char* d_warpInfo;
	cudaMalloc(&d_warpInfo, totalThreads * sizeof(char));
	cudaMemset(d_warpInfo, 1, totalThreads * sizeof(char));

	std::vector<float> lBorderVel(totalThreads);
	std::vector<float> wBorderVel(totalThreads);
	std::vector<float> hBorderVel(totalThreads);

	for(int i = 0; i < totalThreads; i++)
	{
		lBorderVel[i] = xArea[i] > 0 ? VelFlux : 0.0;
	}

	float* xVel, * yVel, * zVel;

		cudaMalloc(&xVel, totalThreads * sizeof(float));
		cudaMalloc(&yVel, totalThreads * sizeof(float));
		cudaMalloc(&zVel, totalThreads * sizeof(float));

		cudaMemcpy(xVel, lBorderVel.data(), totalThreads * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(yVel, 0, totalThreads * sizeof(float));
		cudaMemset(zVel, 0, totalThreads * sizeof(float));


	//valores de entrada dos cubos do volume de controle
	float areaFlux = dyzThreads;

	double totalTimeTeorical = 0.0;
	double totalTimeReal = 0.0;

	//quantidade de energia é preservada por segundo
	
	float instDamping = pow(damping, deltaTime);

	int iter = 0;
	double start = now();
	int lastPercent = -1;

	while (iter <= maxIter)
	{

		fluidMovement <<<blocksDim, threadsDim >>> (
			xVel,
			yVel,
			zVel,
			d_xArea,
			d_yArea,
			d_zArea,
			d_mass,
			d_volume,
			d_warpInfo,
			deltaTime,
			VelFlux,
			areaFlux,
			xThreads,
			yThreads,
			zThreads,
			(int)numThreads);
		//cudaError_t err = cudaGetLastError();
		//printf("Launch error: %s\n", cudaGetErrorString(err));
		checkCuda(cudaDeviceSynchronize(), "fluidMovement");

		recalculateVelocities <<<blocksDim, threadsDim >>> (
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
			(int)numThreads);
		//err = cudaGetLastError();
		//printf("Launch error: %s\n", cudaGetErrorString(err));
		//checkCuda(cudaDeviceSynchronize(), "recalculateVelocities");

		checkCuda(cudaDeviceSynchronize(), "recalculateVelocities");
		totalTimeTeorical += deltaTime;
		iter++;

		int percent = (int)(100.0 * iter / maxIter);
		if (interativo && percent != lastPercent)
		{
			double remainTime = (percent > 0) ? (100 - percent) * (now() - start) / percent : 0.0;
			printf("\rProgresso: %3d%% (%d/%d iteracoes) - tempo restante: %.1fs   ", percent, iter, (int)maxIter, remainTime);
			fflush(stdout);
			lastPercent = percent;
		}
	}

	totalTimeReal += now() - start;
	//lastPrint = floor(totalTimeTeorical);

	if (interativo) printf("\n");

	// traz tudo do device de volta para o host
	cudaMemcpy(warpInfo.data(), d_warpInfo, totalThreads * sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(mass.data(), d_mass, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

	int invalidSimulation = 0;

	if(write)
	{
		cudaMemcpy(volume.data(), d_volume, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(xArea.data(), d_xArea, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(yArea.data(), d_yArea, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(zArea.data(), d_zArea, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

		cudaMemcpy(lBorderVel.data(), xVel, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(wBorderVel.data(), yVel, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hBorderVel.data(), zVel, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
	}

	float skippedWarps = 0.0f;
	for(int c = 0; c < totalThreads; c++)
	{
		skippedWarps += warpInfo[c];
		invalidSimulation ^= (mass[c] < 0);
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

			auto idx = [](int x, int y, int z) -> size_t
			{
				const size_t sizeBlock = (size_t)nxThreads * nyThreads * nzThreads;

				const size_t bloco = (size_t)(x / nxThreads)
				                   + (size_t)nxBlock * ((size_t)(y / nyThreads)
				                   + (size_t)nyBlock * (size_t)(z / nzThreads));

				const size_t thread = (size_t)(x % nxThreads)
				                    + (size_t)nxThreads * ((size_t)(y % nyThreads)
				                    + (size_t)nyThreads * (size_t)(z % nzThreads));

				return sizeBlock * bloco + thread;
			};

			for (int z = 0; z < zThreads; z++)
			{
				for (int y = 0; y < yThreads; y++)
				{
					for (int x = 0; x < xThreads; x++)
					{
						const size_t k = idx(x, y, z);

						float density = (volume[k] != 0.0) ? mass[k] / volume[k] : 0.0;

						fprintf(dataFile, "[%zu] (x=%d y=%d z=%d)  mass=%.4lf  volume=%.4f  density=%.4f  cubos=%d  warpskip=%d  "
							"xArea=%.4f  yArea=%.4f  zArea=%.4f  "
							"xVel=%.4lf  yVel=%.4lf  zVel=%.4lf\n",
							k, x, y, z,
							mass[k], volume[k], density, (int)cubos[k], (int)warpInfo[k],
							xArea[k], yArea[k], zArea[k],
							lBorderVel[k], wBorderVel[k], hBorderVel[k]);
					}
				}
			}
		}
		
		fprintf(dataFile, "\n");
		fclose(dataFile);
	}
	else
	{
		fprintf(stderr, "Erro ao abrir %s para escrita\n", filename);
	}

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