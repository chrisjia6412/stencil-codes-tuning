#include "../host/inc/convolution.h"

__kernel 
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int out_radius)
{

    float shift_register[KERNEL_SIZE+7];
    // initialize the shift array
    for (int i = 0; i < KERNEL_SIZE+6; i++) {
        shift_register[i] = 0.0f;
    }

    for (int x = 0; x < out_radius; x+=8) {
        // shift register implementation
        #pragma unroll
        for (int i = 0; i < KERNEL_SIZE-1; i++) {
            shift_register[i] = shift_register[i+8];
        }
        shift_register[KERNEL_SIZE-1] = srcMatrix[x];
        shift_register[KERNEL_SIZE] = srcMatrix[x+1];
        shift_register[KERNEL_SIZE+1] = srcMatrix[x+2];
        shift_register[KERNEL_SIZE+2] = srcMatrix[x+3];
        shift_register[KERNEL_SIZE+3] = srcMatrix[x+4];
        shift_register[KERNEL_SIZE+4] = srcMatrix[x+5];
        shift_register[KERNEL_SIZE+5] = srcMatrix[x+6];
        shift_register[KERNEL_SIZE+6] = srcMatrix[x+7];

        float running_sum_0 = 0.0f;
        float running_sum_1 = 0.0f;
        float running_sum_2 = 0.0f;
        float running_sum_3 = 0.0f;
        float running_sum_4 = 0.0f;
        float running_sum_5 = 0.0f;
        float running_sum_6 = 0.0f;
        float running_sum_7 = 0.0f;

        #pragma unroll
        for (int a = 0; a < KERNEL_SIZE; a++) {
            running_sum_0 += shift_register[KERNEL_SIZE-1-a] * filterMatrix[a];
            running_sum_1 += shift_register[KERNEL_SIZE-a] * filterMatrix[a];
            running_sum_2 += shift_register[KERNEL_SIZE+1-a] * filterMatrix[a];
            running_sum_3 += shift_register[KERNEL_SIZE+2-a] * filterMatrix[a];
            running_sum_4 += shift_register[KERNEL_SIZE+3-a] * filterMatrix[a];
            running_sum_5 += shift_register[KERNEL_SIZE+4-a] * filterMatrix[a];
            running_sum_6 += shift_register[KERNEL_SIZE+5-a] * filterMatrix[a];
            running_sum_7 += shift_register[KERNEL_SIZE+6-a] * filterMatrix[a];

        }
        outMatrix[x] = running_sum_0;  
        outMatrix[x+1] = running_sum_1;   
        outMatrix[x+2] = running_sum_2;   
        outMatrix[x+3] = running_sum_3;   
        outMatrix[x+4] = running_sum_4;   
        outMatrix[x+5] = running_sum_5;   
        outMatrix[x+6] = running_sum_6;   
        outMatrix[x+7] = running_sum_7;   
    }

}
