#pragma once
#include <curand_kernel.h>
#include "payoffs.cuh"

static __global__ void init_curand_kernel(curandState *state, unsigned long seed, int nSim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nSim) {
        curand_init(seed, tid, 0, &state[tid]);
    }
}

template<typename CallPut>
__global__ void monte_carlo_euro(float S0,float *payoffs, float K, float r,float var,float T,float k, float theta,float rho,float xi ,unsigned int nSim,unsigned int nSteps, curandState *state){
    int tid= blockIdx.x*blockDim.x+ threadIdx.x;

    if(tid<nSim){
        curandState localstate=state[tid];
        float dt=T/(float)nSteps;
        float S=S0;
        float v=var;

        CallPut payoff;

        for(int i=0;i<nSteps;i++){
            float dW1=curand_normal(&localstate);
            float dW2=rho*dW1+sqrtf(1.0f-rho*rho)*curand_normal(&localstate);
            float vpos=fmaxf(v,0.0f);
            v=v+k*(theta-vpos)*dt+dW2*sqrtf(vpos)*sqrtf(dt)*xi;
            v=fmaxf(v,0.0f);
            S=S*expf((r-0.5f*v)*dt+sqrtf(v)*sqrtf(dt)*dW1);

        }

        payoffs[tid]=payoff(S,K);
        state[tid]=localstate;
    }
}

template<typename CallPut>
__global__ void monte_carlo_asia(float S0,float *payoffs, float K, float r,float var,float T,float k, float theta,float rho,float xi ,unsigned int nSim,unsigned int nSteps, curandState *state){
    int tid= blockIdx.x*blockDim.x+ threadIdx.x;

    if(tid<nSim){
        curandState localstate=state[tid];
        float dt=T/(float)nSteps;
        float S=S0;
        float v=var;

        CallPut payoff;
        double sum=S0;
        for(int i=0;i<nSteps;i++){
            float dW1=curand_normal(&localstate);
            float dW2=rho*dW1+sqrtf(1.0f-rho*rho)*curand_normal(&localstate);
            float vpos=fmaxf(v,0.0f);
            v=v+k*(theta-vpos)*dt+dW2*sqrtf(vpos)*sqrtf(dt)*xi;
            v=fmaxf(v,0.0f);
            S=S*expf((r-0.5f*v)*dt+sqrtf(v)*sqrtf(dt)*dW1);
            sum+=S;
        }
        float S_f= sum/(nSteps+1);
        payoffs[tid]=payoff(S_f,K);
        state[tid]=localstate;
    }
}

template<typename CallPut>
__global__ void monte_carlo_lookback(float S0,float *payoffs, float K, float r,float var,float T,float k, float theta,float rho,float xi ,unsigned int nSim,unsigned int nSteps, curandState *state){
    int tid= blockIdx.x*blockDim.x+ threadIdx.x;

    if(tid<nSim){
        curandState localstate=state[tid];
        float dt=T/(float)nSteps;
        float S=S0;
        float v=var;

        CallPut payoff;
        float minn=S;
        float maxx=S;
        for(int i=0;i<nSteps;i++){
            float dW1=curand_normal(&localstate);
            float dW2=rho*dW1+sqrtf(1.0f-rho*rho)*curand_normal(&localstate);
            float vpos=fmaxf(v,0.0f);
            v=v+k*(theta-vpos)*dt+dW2*sqrtf(vpos)*sqrtf(dt)*xi;
            v=fmaxf(v,0.0f);
            S=S*expf((r-0.5f*v)*dt+sqrtf(v)*sqrtf(dt)*dW1);
            minn=fminf(minn,S);
            maxx=fmaxf(maxx,S);
        }
        
        if constexpr(std::is_same_v<CallPut, Call>){
            payoffs[tid]=payoff(maxx,K);
        }
        else if constexpr(std::is_same_v<CallPut , Put>){
            payoffs[tid]=payoff(minn,K);
        }
        state[tid]=localstate;
    }
}
