// the key idea is to initialize a shift register to cache the srcMatrix
// which is an efficient hardware compared with banked local memory


#include "../host/inc/convolution.h"
// sr size = kernel_size * input_radius + kernel_size - 1
#define SR_SIZE 8266

__kernel 
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int out_radius)
{

    float shift_register[SR_SIZE];
    // init shift register values
    #pragma unroll
    for (int i = 0; i < SR_SIZE; i++) {
        shift_register[i] = 0.0f;
    }

    for (int y = 0; y < out_radius; y++) {
        for (int x = 0; x < out_radius; x+=2) {
            float running_sum_0 = 0.0f;
            float running_sum_1 = 0.0f;
            // shift register implementation starts
            #pragma unroll
            for (int i = SR_SIZE-1; i > 1; i--) {
                shift_register[i] = shift_register[i-2];
            }
            shift_register[1] = srcMatrix[y*1032+x];
            shift_register[0] = srcMatrix[y*1032+x+1];
            // shift register implementation ends

            // unroll all the two level loops
            #pragma unroll
            for (int a = 0; a < KERNEL_SIZE; a++) {
                #pragma unroll
                for (int b = 0; b < KERNEL_SIZE; b++) {
                    running_sum_0 += shift_register[a*1032+b] * filterMatrix[a * KERNEL_SIZE + b];
                    running_sum_1 += shift_register[a*1032+b+1] * filterMatrix[a * KERNEL_SIZE + b];                   
                }
            }
            if((x>=8) && (y>=8)) {
                outMatrix[(y-8) * 1024 + x - 8] = running_sum_1;
                outMatrix[(y-8) * 1024 + x - 7] = running_sum_0;
            }
        }
    }

}
