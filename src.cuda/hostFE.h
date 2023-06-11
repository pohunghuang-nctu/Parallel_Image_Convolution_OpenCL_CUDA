#ifndef __HOSTFE__
#define __HOSTFE__
// #define CL_TARGET_OPENCL_VERSION 220
// #include <CL/cl.h>
#ifdef __cplusplus
extern "C" {
#endif

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage);

#ifdef __cplusplus
}
#endif
#endif

