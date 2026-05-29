# CUDA Monte Carlo Option Pricer (Heston Model)

A high-performance, GPU-accelerated Monte Carlo option pricing engine implemented in CUDA C++. This project prices various financial derivatives under the Heston stochastic volatility model.

## Features

* **Supported Options**: European, Asian, and Lookback options.
* **Supported Payoffs**: Call and Put.
* **Underlying Model**: Heston stochastic volatility model (Euler-Maruyama discretization).
* **GPU Acceleration**: Utilizes CUDA for massively parallel path generation and pricing, providing up to 250x+ speedup over single-threaded CPU implementations.
* **Random Number Generation**: Employs `curand` for fast, parallel pseudo-random number generation on the GPU.
* **Custom Reduction**: Implements a highly optimized, custom multi-block grid-stride reduction to efficiently compute the final option price without memory bottlenecks.
* **CPU Benchmark**: Includes a complete CPU-based implementation for accuracy verification and performance benchmarking.

## Prerequisites

* NVIDIA GPU with CUDA architecture support.
* CUDA Toolkit 13.0 installed (`nvcc` compiler available in your PATH).
* C++17 compatible host compiler.

## Project Structure

* `src/`: Contains the main entry point (`main.cu`), core runner logic (`run.cuh`), and template instantiations (`instantiate.cu`).
* `include/options/`: Contains the core CUDA kernels for the simulation (`simulator.cuh`), payoffs (`payoffs.cuh`), and the high-performance reduction logic (`sum_reduction.cuh`).
* `include/cpu/`: Contains the CPU implementations used for benchmarking and validation.
* `include/input/`: Handles the interactive command-line interface for parameter input.
* `include/benchmark/`: Contains timing utilities for performance measurement.

## Compilation

Navigate to the root directory of the project and compile the source files using `nvcc`. The `-O3` flag is highly recommended to enable maximum compiler optimizations.

```bash
nvcc -O3 src/main.cu src/instantiate.cu -o src/main
```
or
```bash
nvcc -O3 -std=c++17 src/main.cu src/instantiate.cu -o src/main
```

## How to Run

Execute the compiled binary from the command line:

```bash
./src/main
```

Upon running, the program will present an interactive prompt allowing you to configure the simulation. You can either enter your custom parameters or press `Enter` to use the default values.

### Interactive Parameters

The program allows you to configure the following parameters interactively:

1. **Compare with CPU**: Option to run the single-threaded CPU simulation alongside the GPU for benchmarking and validation.
2. **Option Type**: European, Asian, or Lookback.
3. **Payoff Type**: Call or Put.
4. **Market Parameters**:
    * Spot Price (S0)
    * Strike Price (K)
    * Time to Maturity in days (T)
    * Risk-free Interest Rate (r)
5. **Heston Model Parameters**:
    * Initial Variance (v0)
    * Mean Reversion Rate (k)
    * Terminal/Long-run Variance (theta)
    * Correlation Factor (rho)
    * Volatility of Variance (xi)
6. **Simulation Parameters**:
    * Number of Monte Carlo Paths/Simulations
    * Number of Time Steps

### Performance Results (1,000,000 Paths, 1,000 Steps)

The following table presents the execution metrics comparing the GPU simulation (using our custom multi-block reduction) to the single-threaded CPU reference implementation:

| Metric | GPU | CPU (Single-Threaded) | Speedup / Improvement |
| :--- | :--- | :--- | :--- |
| **RNG Initialization** | 32.00 ms | - | - |
| **Simulation Path Run** | 25.95 ms | - | - |
| **Total Price Execution** | 180.83 ms | 47,253.50 ms (~47.25 seconds) | **~261x Speedup** |
| **Option Price Output** | 83.4765 | 83.5078 | **0.037% Difference** |

## Benchmarking

The program prints detailed execution times and performance throughput (in millions of paths per second) for the following phases:
1. **RNG Initialization**: Time spent generating random states with `curand_init`.
2. **Simulation**: Time spent generating Monte Carlo paths on the GPU/CPU.
3. **Total Time**: Time taken for memory allocations, kernel launches, reduction, and transferring results.

### Running a Benchmark Check

To run a performance check on your system, execute the binary and configure it as follows:

1. Enable the CPU comparison by entering `1` for the first prompt.
2. Select options of your choice.
3. Configure the number of simulations to example `1000000` (1 million).
4. View the speedup factor between GPU and CPU execution.

For 1 million paths, the GPU pricing engine typically completes in under **200 milliseconds** (achieving a **~250x speedup** compared to single-threaded CPU implementations).
