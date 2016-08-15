#include "../host/inc/convolution.h"

__kernel 
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int out_radius)
{

    float shift_register[KERNEL_SIZE];
    // initialize the shift array
    for (int i = 0; i < KERNEL_SIZE; i++) {
        shift_register[i] = 0.0f;
    }

    for (int x = 0; x < out_radius; x++) {
        // shift register implementation
        #pragma unroll
        for (int i = 0; i < KERNEL_SIZE-1; i++) {
            shift_register[i] = shift_register[i+1];
        }
        shift_register[KERNEL_SIZE-1] = srcMatrix[x];

        float running_sum = 0.0f;
        #pragma unroll
        for (int a = 0; a < KERNEL_SIZE; a++) {
            running_sum += shift_register[KERNEL_SIZE-1-a] * filterMatrix[a];
        }
        outMatrix[x] = running_sum;   
    }

}
