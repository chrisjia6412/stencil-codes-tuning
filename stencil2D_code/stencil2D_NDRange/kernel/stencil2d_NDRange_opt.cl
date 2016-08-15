#include "../host/inc/stencil.h"

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
__attribute((num_compute_units(COMPUTE_UNITS)))
void stencil(__global float *restrict outMatrix,
             __global float *restrict srcMatrix, 
             int src_radius)
{
    // local memory
    __local float src_local[BLOCK_SIZE+2][BLOCK_SIZE+2];


    // Block index
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);

    // Local ID index (offset within a block)
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int block_start_x = BLOCK_SIZE * block_x;
    int block_start_y = BLOCK_SIZE * block_y;

    float running_sum = 0.0f;

    // 16*16 work items fetch 18*18 elements, only 9*9 work-items are active in this stage
    if(local_x < 9 && local_y < 9) {
        src_local[2*local_y][2*local_x] = srcMatrix[(block_start_y + 2 * local_y) * src_radius + block_start_x + 2 * local_x];
        src_local[2*local_y][2*local_x+1] = srcMatrix[(block_start_y + 2 * local_y) * src_radius + block_start_x + 2 * local_x + 1];
        src_local[2*local_y+1][2*local_x] = srcMatrix[(block_start_y + 2 * local_y + 1) * src_radius + block_start_x + 2 * local_x];
        src_local[2*local_y+1][2*local_x+1] = srcMatrix[(block_start_y + 2 * local_y + 1) * src_radius + block_start_x + 2 * local_x + 1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);


    running_sum = (src_local[local_y+1][local_x] + src_local[local_y+1][local_x+1] + src_local[local_y+1][local_x+2] + src_local[local_y][local_x+1] + src_local[local_y+2][local_x+1]) * 0.2;

    // Store result in matrix C
    outMatrix[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}
