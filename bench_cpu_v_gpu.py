# pytorch_benchmark.py
import torch
import time

def benchmark(device_name, size=8192, runs=3):
    """
    Runs a matrix multiplication benchmark on a specified PyTorch device.

    Args:
        device_name (str): The name of the device to run on ('cpu', 'cuda', 'mps').
        size (int): The size of the square matrices to be multiplied (size x size).
        runs (int): The number of times to run the operation for averaging.

    Returns:
        float: The average time in seconds for one matrix multiplication, or None on error.
    """
    try:
        device = torch.device(device_name)
        # Ensure size is an integer for tensor dimensions
        matrix_size = int(size)

        print(f"\n--- Benchmarking on: {device_name} ---")
        print(f"Matrix size: {matrix_size}x{matrix_size}")

        # Create two large random tensors on the specified device
        # FIX: Pass dimensions as separate integer arguments, not a tuple.
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)

        # Warm-up runs to handle any initial overhead
        print("Warming up...")
        for _ in range(5):
            _ = a @ b

        # Synchronize to ensure warm-up is complete before starting the timer
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

        # Main benchmark loop
        print(f"Running benchmark ({runs} runs)...")
        times = []
        for _ in range(runs):
            start_time = time.time()
            _ = a @ b
            # Synchronize to get accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / runs
        print(f"Average time over {runs} runs: {avg_time:.4f} seconds")
        return avg_time

    except Exception as e:
        print(f"An error occurred on device {device_name}: {e}")
        return None

def main():
    """
    Main function to detect devices and run the benchmarks.
    """
    print(f"PyTorch Version: {torch.__version__}")

    # --- Device Detection ---
    gpu_device_name = None
    if torch.cuda.is_available():
        gpu_device_name = "cuda"
        print(f"Detected NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_device_name = "mps"
        print("Detected Apple MPS GPU.")
    else:
        print("No compatible GPU found (CUDA or MPS).")

    # --- Run Benchmarks ---
    cpu_time = benchmark("cpu")

    if gpu_device_name and cpu_time is not None:
        gpu_time = benchmark(gpu_device_name)
        if gpu_time is not None and gpu_time > 0:
            print("\n--- Comparison ---")
            print(f"CPU Time: {cpu_time:.4f}s")
            print(f"GPU Time: {gpu_time:.4f}s")
            print(f"GPU is {cpu_time / gpu_time:.1f}x faster than CPU.")
    else:
        print("\nNo GPU available to compare against.")

if __name__ == "__main__":
    main()
