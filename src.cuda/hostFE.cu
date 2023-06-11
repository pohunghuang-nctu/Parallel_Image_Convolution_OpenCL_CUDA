#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
//#include "helper.h"
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float F3[9];
__constant__ float F5[25];
__constant__ float F7[49];

__global__ void convolution(const float* inputImage, float* outputImage, 
                            int imageHeight, int imageWidth, 
                            int filterWidth, int pitch_width_in, int pitch_width_out)
{
    const float *filter;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (filterWidth == 3) {
        filter = F3;
    } else if (filterWidth == 5) {
        filter = F5;
    } else if (filterWidth == 7) {
        filter = F7;
    } else  {}

    int halffilterSize = filterWidth / 2;
    float sum = 0;

    if(i < imageHeight && j < imageWidth) {
        for (int k = -halffilterSize; k <= halffilterSize; k++)
        {
            for (int l = -halffilterSize; l <= halffilterSize; l++)
            {
                if (i + k >= 0 && i + k < imageHeight &&
                    j + l >= 0 && j + l < imageWidth)
                {
                    sum += inputImage[(i + k) * pitch_width_in + j + l] *
                           filter[(k + halffilterSize) * filterWidth +
                                  l + halffilterSize];
                }
            }
        }
        outputImage[i * pitch_width_out + j] = sum;
    }
}

extern "C" void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage)
{


    int filterSize = filterWidth * filterWidth;

    if (filterWidth == 3)
    {
        cudaMemcpyToSymbol(F3, filter, filterSize * sizeof(float));
    }
    else if (filterWidth == 5)
    {
        cudaMemcpyToSymbol(F5, filter, filterSize * sizeof(float));
    }
    else if (filterWidth == 7)
    {
        cudaMemcpyToSymbol(F7, filter, filterSize * sizeof(float));
    }
    else
    {
        printf("Filter width not supported\n");
        exit(1);
    }

    // Allocate GPU buffers for three vectors (two input, one output)
    float *dev_input, *dev_filter, *dev_output;
    size_t pitch_in, pitch_out;
    int dev_width_in, dev_width_out;
    
    // input memory allocation
    cudaMallocPitch((void**)&dev_input, &pitch_in, imageWidth * sizeof(float), imageHeight);
    dev_width_in = pitch_in / sizeof(float);
    printf("input pitch: %zu, width: %d\n", pitch_in, dev_width_in);
    fflush(stdout);
    // filter memory allocation
    cudaMalloc((void**)&dev_filter, filterSize * sizeof(float));

    // output memory allocation
    cudaMallocPitch((void**)&dev_output, &pitch_out, imageWidth * sizeof(float), imageHeight);
    dev_width_out = pitch_out / sizeof(float);
    printf("output pitch: %zu, width: %d\n", pitch_out, dev_width_out);
    fflush(stdout);    

    // Copy input array data to GPU
    cudaMemcpy2D(dev_input, pitch_in, inputImage, imageWidth * sizeof(float), imageWidth * sizeof(float), imageHeight, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_filter, filter, filterSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 dimBlock(16, 16); // block size, 25x25 threads per block
    dim3 dimGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0)); // grid size, depending on the image size and block size

    // Launch the Kernel on the GPU

    convolution<<<dimGrid, dimBlock>>>(dev_input, dev_output, imageHeight, imageWidth, filterWidth, dev_width_in, dev_width_out);

    // CUDA Device Synchronize 
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy2D(outputImage, imageWidth * sizeof(float), dev_output, pitch_out, imageWidth * sizeof(float), imageHeight, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(dev_input);
    cudaFree(dev_filter);
    cudaFree(dev_output);
}
