#include "../host/inc/demystify.h"

__kernel 
void demystify(__global int *restrict srcMatrix,
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
    int temp_value_8 = 8; 
    int temp_value_9 = 9;  
    int temp_value_10 = 10;  
    int temp_value_11 = 11;  
    int temp_value_12 = 12; 
    int temp_value_13 = 13;  
    int temp_value_14 = 14;  
    int temp_value_15 = 15;  



    __local int local_src[1024]; 

    for(int i=0; i<1024;i++) {
        local_src[i] = srcMatrix[i];
    }

    // loop 1024*1024 times through the array
    #pragma unroll 1
    for(int i=0; i<67108864; i++) {
        temp_value_0 = local_src[temp_value_0];
        temp_value_1 = local_src[temp_value_1];
        temp_value_2 = local_src[temp_value_2];
        temp_value_3 = local_src[temp_value_3];
        temp_value_4 = local_src[temp_value_4];
        temp_value_5 = local_src[temp_value_5];
        temp_value_6 = local_src[temp_value_6];
        temp_value_7 = local_src[temp_value_7];
        temp_value_8 = local_src[temp_value_8];
        temp_value_9 = local_src[temp_value_9];
        temp_value_10 = local_src[temp_value_10];
        temp_value_11 = local_src[temp_value_11];
        temp_value_12 = local_src[temp_value_12];
        temp_value_13 = local_src[temp_value_13];
        temp_value_14 = local_src[temp_value_14];
        temp_value_15 = local_src[temp_value_15];


    } 

    outMatrix[0] = temp_value_0;
    outMatrix[1] = temp_value_1;
    outMatrix[2] = temp_value_2;
    outMatrix[3] = temp_value_3;
    outMatrix[4] = temp_value_4;
    outMatrix[5] = temp_value_5;
    outMatrix[6] = temp_value_6;
    outMatrix[7] = temp_value_7;
    outMatrix[8] = temp_value_8;
    outMatrix[9] = temp_value_9;
    outMatrix[10] = temp_value_10;
    outMatrix[11] = temp_value_11;
    outMatrix[12] = temp_value_12;
    outMatrix[13] = temp_value_13;
    outMatrix[14] = temp_value_14;
    outMatrix[15] = temp_value_15;

}
