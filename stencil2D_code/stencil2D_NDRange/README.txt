Here are the NDRange naive kernel and optimized kernel for stencil_2D.

optimized kernel include local Memory (18*18) + SIMD (4) + Compute Unit Replication (8). The input data size is 1024*1024.

Use the command below to build the kernel codes:

(1) for naive kernel: 
    aoc kernel_file.cl --fpc --fp-relaxed --report
(2) for opt kernel: aoc
    aoc kernel_file.cl --fpc --fp-relaxed -DSIMD_WORK_ITEMS=4 -DCOMPUTE_UNITS=8 --report

Use the command below to build and run the host codes:

    make clean
    make
    bin/exe_file 1024 1024
