#include "../host/inc/stencil.h"

__kernel 
void stencil(__global float *restrict outMatrix,
             __global float *restrict srcMatrix, 
             int input_radius,
             int out_radius)
{
    #pragma unroll 1
    for (int y = 0; y < out_radius; y++) {
        #pragma unroll 1
        for (int x = 0; x < out_radius; x++) {
            outMatrix[y*out_radius+x] = 0.2*(srcMatrix[(y+1)*input_radius+x+1] + srcMatrix[y*input_radius+x+1] + srcMatrix[(y+2)*input_radius+x+1] + srcMatrix[(y+1)*input_radius+x] + srcMatrix[(y+1)*input_radius+x+2]);    
        }
    }

}
