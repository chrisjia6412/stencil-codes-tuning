#include "../host/inc/convolution.h"

__kernel 
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int src_radius,
                 int out_radius)
{

    for (int y = 0; y < out_radius; y++) {
        for (int x = 0; x < out_radius; x++) {
            float running_sum = 0.0f;
            for (int a = 0; a < KERNEL_SIZE; a++) {
                for (int b = 0; b < KERNEL_SIZE; b++) {
                    running_sum += srcMatrix[(y + 8  - a) * src_radius + x + 8 - b] * filterMatrix[a * KERNEL_SIZE + b];
                }
            }
            outMatrix[y * out_radius + x] = running_sum;
        }
    }

}
