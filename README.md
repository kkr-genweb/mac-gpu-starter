# PyTorch GPU vs. CPU Benchmark

This script provides a simple benchmark for comparing the performance of matrix multiplication on a GPU versus a CPU using PyTorch. It automatically detects the most appropriate available GPU device (NVIDIA CUDA or Apple Metal) and runs the benchmark on it.

## Description

The script performs the following steps:
1. Checks for the availability of an NVIDIA GPU (via CUDA) or an Apple Silicon GPU (via Metal Performance Shaders - MPS).
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
% uv run bench_cpu_v_gpu.py

PyTorch Version: 2.7.1
Detected Apple MPS GPU.

--- Benchmarking on: cpu ---
Matrix size: 8192x8192
Warming up...
Running benchmark (3 runs)...
Average time over 3 runs: 0.3931 seconds

--- Benchmarking on: mps ---
Matrix size: 8192x8192
Warming up...
Running benchmark (3 runs)...
Average time over 3 runs: 0.1266 seconds

--- Comparison ---
CPU Time: 0.3931s
GPU Time: 0.1266s
GPU is 3.1x faster than CPU.
```