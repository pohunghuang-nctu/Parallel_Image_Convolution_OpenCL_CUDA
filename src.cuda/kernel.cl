#define CL_TARGET_OPENCL_VERSION 220
//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#pragma OPENCL EXTENSION cl_khr_printf : enable

__kernel void convolution(
    const __global float* inputImage, 
    __global float* outputImage, 
    __constant float* filter, 
    int imageHeight, 
    int imageWidth, 
    int filterWidth)
{
    int i = get_global_id(1);
    //printf("global size 0: %d\n", get_global_size(0));
    int j = get_global_id(0);

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
                    sum += inputImage[(i + k) * imageWidth + j + l] *
                           filter[(k + halffilterSize) * filterWidth +
                                  l + halffilterSize];
                }
            }
        }
        outputImage[i * imageWidth + j] = sum;
    }
}
