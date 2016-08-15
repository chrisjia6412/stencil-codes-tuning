Here are the SingleTask naive kernel and optimized kernel for convolution_2D.

optimized kernel include Shift Register Pattern + Task Coalescing (1). The input data size is 1024*1024 and the filter data size is 9*9.

Use the command below to build the kernel codes:

    aoc kernel_file.cl --fpc --fp-relaxed --report

Use the command below to build and run the host codes:

    make clean
    make
    bin/exe_file 

    For naive kernel, use main_naive.cpp as host codes. For the other kernels, use main_SRP.cpp as host codes.
