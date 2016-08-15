#include "../host/inc/demystify.h"

__kernel 
void demystify(__constant int *srcMatrix,
               __global int *restrict outMatrix)
{  
    int temp_value_0 = get_local_id(0);
    int temp_value_1 = 1;
    int temp_value_2 = 2;
    int temp_value_3 = 3;
    int temp_value_4 = 4;
    int temp_value_5 = 5;
    int temp_value_6 = 6;
    int temp_value_7 = 7;

    // loop 1024*1024 times through the array
    #pragma unroll 1
    for(int i=0; i<67108864; i++) {
        temp_value_0 = srcMatrix[temp_value_0];
        temp_value_1 = srcMatrix[temp_value_1];
  	temp_value_2 = srcMatrix[temp_value_2];
  	temp_value_3 = srcMatrix[temp_value_3];
  	temp_value_4 = srcMatrix[temp_value_4];
  	temp_value_5 = srcMatrix[temp_value_5];
  	temp_value_6 = srcMatrix[temp_value_6];
  	temp_value_7 = srcMatrix[temp_value_7];

    } 

    outMatrix[0] = temp_value_0;
    outMatrix[1] = temp_value_1;
    outMatrix[2] = temp_value_2;
    outMatrix[3] = temp_value_3;
    outMatrix[4] = temp_value_4;
    outMatrix[5] = temp_value_5;
    outMatrix[6] = temp_value_6;
    outMatrix[7] = temp_value_7;
}
