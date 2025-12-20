#pragma once
#include<iostream>
#include "../include/options/options.cuh"
#include "../include/benchmark/benchmark.cuh"

template <typename OptionType>
void run_gpu(float S0, float K,float T,float r,float v0,float k, float theta,float rho,float xi,int nSim,int nSteps){
    Timer timer;
    timer.start();

    int block=256;
    int grid=(nSim+block-1)/block;

    curandState *d_state;
    cudaMalloc(&d_state,nSim*sizeof(curandState));

    CudaBenchmark bench_rng;
    bench_rng.start();

    init_curand_kernel<<<grid,block>>>(d_state,1234ULL,nSim);
    cudaDeviceSynchronize();

    bench_rng.stop();
    bench_rng.report("RNG",nSim);
    
    float *d_payoffs;
    cudaMalloc(&d_payoffs, nSim*sizeof(float));

    OptionType Option(S0,K,T,r,v0,k,theta,rho,xi); 

    CudaBenchmark bench_sim;
    bench_sim.start();

    Option.simulate(d_payoffs,nSim,nSteps,d_state);

    bench_sim.stop();
    bench_sim.report("Simulation", nSim);
    
    float price=Option.calculate(d_payoffs,nSim,r,T);

    cudaFree(d_state); cudaFree(d_payoffs);
    
    timer.stop();
    timer.report();

    std::cout<< "Option Price: " << price << std::endl;
}