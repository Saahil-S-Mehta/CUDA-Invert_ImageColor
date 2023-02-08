
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Inversion_CUDA.h"

__global__ void Inversion_CUDA(unsigned char* Image, int Channels);
__global__ void Gray_CUDA(unsigned char* Image, int Channels);

void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;

	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);

	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Inversion_CUDA << <Grid_Image, 8 >> > (Dev_Input_Image, Channels);

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	cudaFree(Dev_Input_Image);
}

void Image_Gray_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;

	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);

	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Gray_CUDA << <Grid_Image, 1 >> > (Dev_Input_Image, Channels);

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	cudaFree(Dev_Input_Image);
}

__global__ void Gray_CUDA(unsigned char* Image, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * Channels;

	unsigned char temp;

	for (int i = 0; i < Channels; i++) {
		temp += Image[idx + i];
	}

	//temp = temp / 3;

	for (int i = 0; i < Channels; i++) {
		Image[idx + i] = (unsigned char)temp/3;
	}
}

__global__ void Inversion_CUDA(unsigned char* Image, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		Image[idx + i] = 255 - Image[idx + i];
	}
}
