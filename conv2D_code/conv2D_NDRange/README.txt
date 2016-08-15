Here are the NDRange naive kernel and optimized kernel for convolution_2D.

optimized kernel include local Memory (40*40) + Loop Unrolling + SIMD (2). The input data size is 1024*1024 and the filter data size is 9*9.

Use the command below to build the kernel codes:

(1) for naive kernel: 
    aoc kernel_file.cl --fpc --fp-relaxed --report
(2) for opt kernel: aoc
    aoc kernel_file.cl --fpc --fp-relaxed -DSIMD_WORK_ITEMS=2 --report

Use the command below to build and run the host codes:

    make clean
    make
    bin/exe_file 1024 1024
