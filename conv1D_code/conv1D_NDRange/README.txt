Here are the NDRange naive kernel and optimized kernel for convolution_1D.

optimized kernel include local Memory (48) + Loop Unrolling + SIMD (8). The input data size is 1048576*1 and the filter data size is 17*1.

Use the command below to build the kernel codes:

(1) for naive kernel: 
    aoc kernel_file.cl --fpc --fp-relaxed --report
(2) for opt kernel: aoc
    aoc kernel_file.cl --fpc --fp-relaxed -DSIMD_WORK_ITEMS=8 --report

Use the command below to build and run the host codes:

    make clean
    make
    bin/exe_file 
