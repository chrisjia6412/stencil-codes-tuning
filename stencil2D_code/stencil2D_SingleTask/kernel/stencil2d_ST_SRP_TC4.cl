// the key idea is to initialize a shift register to cache the srcMatrix
// which is an efficient hardware compared with banked local memory

// need to use channel here

#include "../host/inc/stencil.h"
// shift register size is equal to 2*W+3, W=1028
// TC4, plus 3
#define SR_SIZE 2062

__kernel 
void stencil(__global float *restrict outMatrix,
             __global float *restrict srcMatrix)
{

    float shift_register[SR_SIZE];
    // init shift register values
    #pragma unroll
    for (int i = 0; i < SR_SIZE; i++) {
        shift_register[i] = 0.0f;
    }

    for (int y = 0; y <1025 ; y++) {
        for (int x = 0; x < 1028; x=x+4) {
            float running_sum_0 = 0.0f;
            float running_sum_1 = 0.0f;
            float running_sum_2 = 0.0f;
            float running_sum_3 = 0.0f;

            // first element
            // shift register implementation starts
            #pragma unroll
            for (int i = SR_SIZE-1; i > 3; i--) {
                shift_register[i] = shift_register[i-4];
            }
            shift_register[0] = srcMatrix[(y+1)*1028+x+3];
            shift_register[1] = srcMatrix[(y+1)*1028+x+2];
            shift_register[2] = srcMatrix[(y+1)*1028+x+1];
            shift_register[3] = srcMatrix[(y+1)*1028+x  ];


            // shift register implementation ends

            // calculation
            running_sum_0 = 0.2*(shift_register[0] + shift_register[1027] + shift_register[1028] + shift_register[1029] + shift_register[2056]);
            running_sum_1 = 0.2*(shift_register[1] + shift_register[1028] + shift_register[1029] + shift_register[1030] + shift_register[2057]);
            running_sum_2 = 0.2*(shift_register[2] + shift_register[1029] + shift_register[1030] + shift_register[1031] + shift_register[2058]);
            running_sum_3 = 0.2*(shift_register[3] + shift_register[1030] + shift_register[1031] + shift_register[1032] + shift_register[2059]);


            outMatrix[y * 1028 + x    ] = running_sum_3;
            outMatrix[y * 1028 + x + 1] = running_sum_2;
            outMatrix[y * 1028 + x + 2] = running_sum_1;
            outMatrix[y * 1028 + x + 3] = running_sum_0;

        }
    }

}
