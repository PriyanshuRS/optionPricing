#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<random>
#include<iostream>
#include "../include/sum_reduction.cuh"
#include "../include/payoffs.cuh"
#include "../include/european.cuh"

int main(){
    float S0    = 101.0f;
    float K     = 100.0f;
    float r     = 0.05f;
    float T     = 1.0f;
    int nSim  = 500000;
    int nSteps = 1000;
    float v0    = 0.01f;
    float theta = 0.04f;
    float k     = 1.5f;
    float xi    = 0.1f;
    float rho   = 0.0f;

    int block=256;
    int grid=(nSim+block-1)/block;

    curandState *d_state;
    cudaMalloc(&d_state,nSim*sizeof(curandState));

    init_curand_kernel<<<grid,block>>>(d_state,1234ULL,nSim);
    cudaDeviceSynchronize();
    
    float *d_payoffs;
    cudaMalloc(&d_payoffs, nSim*sizeof(float));

    European<Call> Option(S0,K,T,r,v0,k,theta,rho,xi); 

    Option.simulate(d_payoffs,nSim,nSteps,d_state);

    float price=Option.calculate(d_payoffs,nSim,r,T);

    cudaFree(d_state); cudaFree(d_payoffs);

    std::cout<< "Option Price: " << price << std::endl;
}