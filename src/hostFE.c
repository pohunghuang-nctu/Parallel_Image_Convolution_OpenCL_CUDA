#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{


    cl_int status;
    // Create command queue first
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    int filterSize = filterWidth * filterWidth;
    // Create OpenCL memory objects for input and output images
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageHeight * imageWidth * sizeof(float), NULL, &status);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, &status);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageHeight * imageWidth * sizeof(float), NULL, &status);
    // Create and build the OpenCL kernel program
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    // set kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputBuffer);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&filterBuffer);    
    status = clSetKernelArg(kernel, 3, sizeof(int), (void *)&imageHeight);
    status = clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageWidth);
    status = clSetKernelArg(kernel, 5, sizeof(int), (void *)&filterWidth);
    // Copy input image data to the input buffer
    status = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, imageHeight * imageWidth * sizeof(float), inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, filterBuffer, CL_TRUE, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);
    // Define global and local work sizes
    size_t globalSize[2] = {imageWidth, imageHeight};
    size_t localSize[2] = {25, 25}; // You can experiment with different local sizes

    // Enqueue the OpenCL kernel for execution
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);

    // Read the output image data from the output buffer
    status = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageHeight * imageWidth * sizeof(float), outputImage, 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(kernel);
}
