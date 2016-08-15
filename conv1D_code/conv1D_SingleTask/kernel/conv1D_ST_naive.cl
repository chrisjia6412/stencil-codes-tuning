#include "../host/inc/convolution.h"

__attribute__((num_compute_units(1)))
__kernel 
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int out_radius)
{

    for (int x = 0; x < out_radius; x++) {
        float running_sum = 0.0f;
        for (int a = 0; a < KERNEL_SIZE; a++) {
            running_sum += srcMatrix[x+KERNEL_SIZE-a-1] * filterMatrix[a];
        }
        outMatrix[x] = running_sum;   
    }

}
