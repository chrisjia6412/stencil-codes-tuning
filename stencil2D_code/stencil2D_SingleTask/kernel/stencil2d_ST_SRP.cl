// the key idea is to initialize a shift register to cache the srcMatrix
// which is an efficient hardware compared with banked local memory

// need to use channel here

#include "../host/inc/stencil.h"
// shift register size is equal to 2*W+3, W=1024+1 
#define SR_SIZE 2053

__kernel 
void stencil(__global float *restrict outMatrix,
             __global float *restrict srcMatrix,
             int out_radius)
{

    float shift_register[SR_SIZE];
    // init shift register values
    #pragma unroll
    for (int i = 0; i < SR_SIZE; i++) {
        shift_register[i] = 0.0f;
    }

    for (int y = 0; y < out_radius-1; y++) {
        for (int x = 0; x < out_radius-1; x++) {
            float running_sum = 0.0f;
            // shift register implementation starts
            #pragma unroll
            for (int i = SR_SIZE-1; i > 0; i--) {
                shift_register[i] = shift_register[i-1];
            }
            shift_register[0] = srcMatrix[(y+1)*1026+x+1];
            // shift register implementation ends

            // calculation
            running_sum = 0.2*(shift_register[1] + shift_register[1026] + shift_register[2051] + shift_register[1025] + shift_register[1027]);
            if(x>=1 && y>=1) {
                outMatrix[(y - 1) * 1024 + x - 1] = running_sum;
            }
        }
    }

}
