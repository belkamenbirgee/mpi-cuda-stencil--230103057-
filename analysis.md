# Cache Line and False Sharing Mechanics in Multi-Core Python

## System Configuration

- GitHub Username: `belkamenbirgee`
- CPU Model: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel
- Architecture: AMD64
- Logical CPU Cores: 8
- Selected CPU Cores: 0, 1, 2, 3
- Cache Line Size: 64 bytes
- Elements per Cache Line: 8
- Number of Threads: 4
- Iterations per Thread: 60,000,000
- Total Intended Updates: 240,000,000

## Q1. What is False Sharing?

False sharing occurs when multiple CPU cores modify different variables that are located in the same cache line. Although the variables are independent, the cache-coherence protocol causes the cache line to move between cores. This creates unnecessary communication and can reduce performance.

In this experiment, four threads use separate counters. With stride 1, the counters are 8 bytes apart and share a 64-byte cache line. With stride 4, counters are 32 bytes apart, so pairs can share cache lines. With stride 8, counters are 64 bytes apart, placing them on separate cache lines. With stride 16, counters are 128 bytes apart, providing additional separation.

## Q2. How Does Cache Line Stride Affect Performance?

The experiment tested strides of 1, 4, 8, and 16.

| Stride | Measured Time (ns) | ns per Intended Update |
|---|---:|---:|
| 1 | 1,682,800 | 0.007012 |
| 4 | 1,539,300 | 0.006414 |
| 8 | 1,255,600 | 0.005232 |
| 16 | 1,534,500 | 0.006394 |

The lowest measured time was observed at stride 8. However, LLVM inspection showed that the shared-counter loop was optimized into a single load, addition, and store. Therefore, these measurements are preliminary and cannot be used as reliable evidence of false-sharing performance.

## Q3. What Is the Purpose of CPU Affinity?

CPU affinity assigns threads to selected processor cores. It helps control thread placement and makes experiments more consistent by reducing unpredictable migration between cores.

In this experiment, four logical cores (0, 1, 2, and 3) were selected using CPU affinity. The system reported that affinity was supported.

## Q4. Why Is a Private Counter Faster or More Efficient?

A private counter allows each thread to update its own local variable instead of repeatedly modifying shared memory. This reduces cache-coherence traffic and avoids false sharing during the main computation.

The private-counter benchmark took 117,921,100 ns, with approximately 0.491338 ns per intended update. Its final counters were all correct at 60,000,000.

This result is a preliminary baseline. A valid comparison requires the shared-counter benchmark to perform actual memory updates in every iteration.

## Q5. What Does LLVM Inspection Show?

LLVM inspection was used to examine the compiled Numba worker and determine whether the loop performed native memory operations.

The generated LLVM contained the following sequence:

- One memory load.
- One addition using the total iteration count.
- One memory store.

The repeated loop had been optimized away. This explains the unusually short shared-counter benchmark times.

Consequently, the current LLVM inspection does not verify repeated native memory updates. The shared-counter implementation must be corrected and benchmarked again before the experiment can be considered complete.

## Conclusion

The experiment successfully verified the system configuration, 64-byte memory alignment, four-thread execution, and final counter values. It also demonstrated the importance of inspecting compiled LLVM code when conducting performance experiments with Numba.

However, the shared-counter loop was optimized by LLVM. Therefore, the current stride measurements are preliminary rather than valid false-sharing measurements. A corrected implementation and new benchmark results are required for a final performance conclusion.