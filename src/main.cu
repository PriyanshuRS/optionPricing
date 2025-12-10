#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<random>
#include<iostream>
#include "../include/options.cuh"

int main(){
    float S0    = 447.84f;
    float K     = 447.5f;
    float r     = 0.0417f;
    float T     = 3.0f/252.0f;
    int nSim  = 500000;
    int nSteps = 1000;
    float v0    = 0.1913f;
    float theta = 0.1654f;
    float k     = 4.0f;
    float xi    = 0.019f;
    float rho   = -0.4f;

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