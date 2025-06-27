# PyTorch GPU vs. CPU Benchmark

This script provides a simple benchmark for comparing the performance of matrix multiplication on a GPU versus a CPU using PyTorch. It automatically detects the most appropriate available GPU device (NVIDIA CUDA, AMD ROCm, or Apple Metal) and runs the benchmark on it.

## Description

The script performs the following steps:
1. Checks for the availability of an NVIDIA GPU (via CUDA), an AMD GPU (via ROCm), or an Apple Silicon GPU (via Metal Performance Shaders - MPS).
2. Selects the appropriate device for benchmarking.
3. Runs a matrix multiplication benchmark on the CPU as a baseline.
4. Runs the same benchmark on the detected GPU device, if one is available.
5. Prints the average execution time for both the CPU and GPU, allowing for a direct performance comparison.   
## Requirements
- 
## Usage
Simply run the script from your terminal:
```
uv run bench_cpu_v_gpu.py
```
## Example Output
The output will vary depending on your hardware.
**On a machine with an NVIDIA GPU:**
```
PyTorch Version: 2.3.1+cu121
Detected GPU: cuda.
---
Benchmarking on: cpu
Average time over 3 runs: 1.8543 seconds
---
Benchmarking on: cuda
Average time over 3 runs: 0.0241 seconds
---
GPU is 76.9x faster than CPU.
```

**On an Apple Silicon Mac:**
```
PyTorch Version: 2.3.1
Detected GPU: mps.
---
Benchmarking on: cpu
Average time over 3 runs: 1.1205 seconds
---
Benchmarking on: mps
Average time over 3 runs: 0.0458 seconds
---
GPU is 24.5x faster than CPU.
```

**On a machine with only a CPU:**
```
PyTorch Version: 2.3.1+cpu
No compatible GPU found (CUDA, MPS, or ROCm).
---
Benchmarking on: cpu
Average time over 3 runs: 5.4321 seconds
---
No GPU to compare against.
```