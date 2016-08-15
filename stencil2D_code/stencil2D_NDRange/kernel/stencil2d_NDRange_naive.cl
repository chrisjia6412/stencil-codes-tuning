#include "../host/inc/stencil.h"

__kernel 
__attribute((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__attribute((num_simd_work_items(SIMD_WORK_ITEMS)))
void stencil(__global float *restrict outMatrix,
             __global float *restrict srcMatrix, 
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

    running_sum = (srcMatrix[(global_y+1)*src_radius+global_x] + srcMatrix[(global_y+1)*src_radius+global_x+1] + srcMatrix[(global_y+1)*src_radius+global_x+2] + srcMatrix[(global_y)*src_radius+global_x+1] + srcMatrix[(global_y+2)*src_radius+global_x+1]) * 0.2;

    // Store result in matrix C
    outMatrix[get_global_id(1) * get_global_size(0) + get_global_id(0)] = running_sum;
}
