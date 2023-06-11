#ifndef __HOSTFE__
#define __HOSTFE__
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program);

#endif