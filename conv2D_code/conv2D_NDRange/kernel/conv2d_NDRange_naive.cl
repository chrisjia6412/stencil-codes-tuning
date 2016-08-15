#include "../host/inc/convolution.h"

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int src_radius)
{

    // Block index
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    // Local ID index (offset within a block)
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    float running_sum = 0.0f;

    

    // loop iteration processes one block of the matrix.
    #pragma unroll 1
    for (int a = 0; a < KERNEL_SIZE; a++)
    {
        #pragma unroll 1
        for (int b = 0; b < KERNEL_SIZE; b++)
        {
            running_sum += srcMatrix[(global_y + 8 - a) * src_radius + global_x + 8 - b] * filterMatrix[a * KERNEL_SIZE + b];
        }

        // Wait for the block to be fully consumed before loading the next
        // block.

    }

    // Store result in matrix C
    outMatrix[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}
