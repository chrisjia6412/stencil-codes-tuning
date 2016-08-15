#include "../host/inc/convolution.h"

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,1,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix)
{

    __local float src_local[BLOCK_SIZE+KERNEL_SIZE-1];

    // Block index
    int block_x = get_group_id(0);
    // Local ID index (offset within a block)
    int local_x = get_local_id(0);
    // Global ID index
    int global_x = get_global_id(0);

    int block_start_x = BLOCK_SIZE * block_x;

    float running_sum = 0.0f;
    
    // preload data into local memory
    if(local_x < 24) {
        src_local[2*local_x] = srcMatrix[block_start_x+ 2*local_x];
        src_local[2*local_x+1] = srcMatrix[block_start_x+ 2*local_x + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int a = 0; a < KERNEL_SIZE; a++)
    {
        running_sum += src_local[local_x+KERNEL_SIZE-1-a] * filterMatrix[a];
    }
    

    // Store result in matrix C
    outMatrix[get_global_id(0)] = running_sum;
}
