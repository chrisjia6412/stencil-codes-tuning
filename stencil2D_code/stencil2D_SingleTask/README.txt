Here are the SingleTask naive kernel and optimized kernel for stencil_2D.

optimized kernel include Shift Register Pattern + Task Coalescing (4). The input data size is 1024*1024.

Use the command below to build the kernel codes:

    aoc kernel_file.cl --fpc --fp-relaxed --report

Use the command below to build and run the host codes:

    make clean
    make
    bin/exe_file 

    For naive kernel, use main_naive.cpp as host codes. For SRP kernel, use main_SRP.cpp as host codes. And use main_SRP_TC4.cpp for SRP_TC4 kernel.
