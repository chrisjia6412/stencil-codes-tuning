Here are the codes to charateraize different memory types on FPGAs.

for constant memory and local memory the codes chracterize the memory latencies under different number of ports.

Points to be careful with:

1. When charaterizing global memory latency, make sure your successive memory access will not hit in one embedded cache line. Or you will get the latency of embedded cache rather than the global memory.

2. When charaterizing constant memory latency, make sure your matrix array size can fit in the constant memory size you set.
