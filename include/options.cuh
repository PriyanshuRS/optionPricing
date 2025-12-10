#pragma once
#include "payoffs.cuh"
#include "sum_reduction.cuh"
#include "simulator.cuh"

template <typename CallPut>
class European{
    private:
        float S0,K,T,s,r,k,theta,rho,xi;
    public:
        European(float _S0, float _K,float _T, float _r, float _s, float _k, float _theta, float _rho, float _xi): S0(_S0),K(_K), T(_T),r(_r),s(_s),k(_k), theta(_theta),rho(_rho), xi(_xi) {}

        void simulate(float *d_payoffs, unsigned int nSim, unsigned int nSteps, curandState *d_states){
            dim3 block(256);
            dim3 grid((nSim+block.x-1)/block.x);

            monte_carlo_euro<CallPut><<<grid,block>>>(S0,d_payoffs,K,r,s,T,k,theta,rho,xi,nSim,nSteps,d_states);
            cudaDeviceSynchronize();    
        }
        
        float calculate(float *d_arr,int n,float r,float T){
            float *d_partial;
            cudaMalloc(&d_partial,n*sizeof(float));

            int block=256;
            int grid=(n+block-1)/block;

            sum_kernel<<<grid,block>>>(d_arr,d_partial,n);
            reduce_kernel<<<1,block, block*sizeof(float)>>>(d_partial,n);
            float sum;
            cudaMemcpy(&sum, d_partial,sizeof(float),cudaMemcpyDeviceToHost);

            cudaFree(d_partial);
            return std::exp(-r*T)*sum/n;
        }
};

template <typename CallPut>
class Asian{
    private:
        float S0,K,T,s,r,k,theta,rho,xi;
    public:
        Asian(float _S0, float _K,float _T, float _r, float _s, float _k, float _theta, float _rho, float _xi): S0(_S0),K(_K), T(_T),r(_r),s(_s),k(_k), theta(_theta),rho(_rho), xi(_xi) {}

        void simulate(float *d_payoffs, unsigned int nSim, unsigned int nSteps, curandState *d_states){
            dim3 block(256);
            dim3 grid((nSim+block.x-1)/block.x);

            monte_carlo_asia<CallPut><<<grid,block>>>(S0,d_payoffs,K,r,s,T,k,theta,rho,xi,nSim,nSteps,d_states);
            cudaDeviceSynchronize();    
        }
        
        float calculate(float *d_arr,int n,float r,float T){
            float *d_partial;
            cudaMalloc(&d_partial,n*sizeof(float));

            int block=256;
            int grid=(n+block-1)/block;

            sum_kernel<<<grid,block>>>(d_arr,d_partial,n);
            reduce_kernel<<<1,block, block*sizeof(float)>>>(d_partial,n);
            float sum;
            cudaMemcpy(&sum, d_partial,sizeof(float),cudaMemcpyDeviceToHost);

            cudaFree(d_partial);
            return std::exp(-r*T)*sum/n;
        }
};

template <typename CallPut>
class Lookback{
    private:
        float S0,K,T,s,r,k,theta,rho,xi;
    public:
        Lookback(float _S0, float _K,float _T, float _r, float _s, float _k, float _theta, float _rho, float _xi): S0(_S0),K(_K), T(_T),r(_r),s(_s),k(_k), theta(_theta),rho(_rho), xi(_xi) {}

        void simulate(float *d_payoffs, unsigned int nSim, unsigned int nSteps, curandState *d_states){
            dim3 block(256);
            dim3 grid((nSim+block.x-1)/block.x);

            monte_carlo_lookback<CallPut><<<grid,block>>>(S0,d_payoffs,K,r,s,T,k,theta,rho,xi,nSim,nSteps,d_states);
            cudaDeviceSynchronize();    
        }
        
        float calculate(float *d_arr,int n,float r,float T){
            float *d_partial;
            cudaMalloc(&d_partial,n*sizeof(float));

            int block=256;
            int grid=(n+block-1)/block;

            sum_kernel<<<grid,block>>>(d_arr,d_partial,n);
            reduce_kernel<<<1,block, block*sizeof(float)>>>(d_partial,n);
            float sum;
            cudaMemcpy(&sum, d_partial,sizeof(float),cudaMemcpyDeviceToHost);

            cudaFree(d_partial);
            return std::exp(-r*T)*sum/n;
        }
};

