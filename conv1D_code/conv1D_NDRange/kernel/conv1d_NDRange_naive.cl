#include "../host/inc/convolution.h"

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix)
{

    // Block index
    int block_x = get_group_id(0);
    // Local ID index (offset within a block)
    int local_x = get_local_id(0);
    // Global ID index
    int global_x = get_global_id(0);

    float running_sum = 0.0f;
    

    // #pragma unroll
    for (int a = 0; a < KERNEL_SIZE; a++)
    {
        running_sum += srcMatrix[global_x+KERNEL_SIZE-1-a] * filterMatrix[a];
    }
    

    // Store result in matrix C
    outMatrix[get_global_id(0)] = running_sum;
}
