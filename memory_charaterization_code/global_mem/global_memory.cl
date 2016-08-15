#include "../host/inc/demystify.h"

__kernel 
void demystify(__global int *restrict srcMatrix,
               __global int *restrict outMatrix)
{  

    int temp_value = get_local_id(0);    

    // loop 1024*1024 times through the array
    #pragma unroll 1
    for(int i=0; i<4096*4096; i++) {
        temp_value = srcMatrix[temp_value];
        outMatrix[i] = temp_value;
        
    } 

}
