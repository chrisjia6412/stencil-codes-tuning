#include "../host/inc/convolution.h"

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void convolution(__global float *restrict outMatrix,
                 __global float *restrict srcMatrix,
                 __global float *restrict filterMatrix, 
                 int src_radius)
{
    // Local storage for a apron + input data
    // the length should be equal to BLOCK_SIZE+KERNEL_SIZE-1
    __local float A_local[BLOCK_SIZE+KERNEL_SIZE-1][BLOCK_SIZE+KERNEL_SIZE-1];
    // each thread calculate one output element

    // Block index
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    // Local ID index (offset within a block)
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    // padding the input matrix in host side with 0, remove if statement in kernel
    int block_start_x = BLOCK_SIZE * block_x + 8;
    int block_start_y = BLOCK_SIZE * block_y + 8;
    int local_mem_start_x = block_start_x - 8;
    int local_mem_start_y = block_start_y - 8;

    float running_sum = 0.0f;

    // preload data into the local memory
    // block 8*8, kernel_size 9*9, each work-item fetch 2 elements in x direction and 2 elements in y direction
    if(local_x < 20 && local_y < 20) {
        A_local[2*local_y][2*local_x] = srcMatrix[(local_mem_start_y + 2 * local_y) * src_radius + local_mem_start_x + 2 * local_x];
        A_local[2*local_y][2*local_x+1] = srcMatrix[(local_mem_start_y + 2 * local_y) * src_radius + local_mem_start_x + 2 * local_x + 1];
        A_local[2*local_y+1][2*local_x] = srcMatrix[(local_mem_start_y + (2 * local_y + 1)) * src_radius + local_mem_start_x + 2 * local_x];
        A_local[2*local_y+1][2*local_x+1] = srcMatrix[(local_mem_start_y + (2 * local_y + 1)) * src_radius + local_mem_start_x + 2 * local_x + 1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);   

    // loop iteration processes one block of the matrix.
    #pragma unroll
    for (int a = 0; a < KERNEL_SIZE; a++)
    {

        #pragma unroll
        for (int b = 0; b < KERNEL_SIZE; b++)
        {
            running_sum += A_local[local_y+8-a][local_x+8-b] * filterMatrix[a * KERNEL_SIZE + b];
        }

        // Wait for the block to be fully consumed before loading the next
        // block.

    }

    // Store result in matrix C
    outMatrix[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}
