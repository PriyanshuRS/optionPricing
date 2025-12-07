#include<cmath>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include<random>
#include<iostream>

__global__ void init_curand_kernel(curandState *state, unsigned long seed, int nSim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nSim) {
        curand_init(seed, tid, 0, &state[tid]);
    }
}


__global__ void european_option_heston(float S0,float *payoffs, float K, float r,float stdev,float T,float k, float theta,float rho,float xi ,int nSim,int nSteps, curandState *state){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;

    if(tid<nSim){
        curandState localstate=state[tid];
        float dt=T/nSteps;
        float S=S0;
        float v=stdev;
        for(int i=0;i<nSteps;i++){
            float dW1=curand_normal(&localstate);
            float dW2=rho*dW1+sqrtf(1.0f-rho*rho)*curand_normal(&localstate);
            float vpos=fmaxf(v,0.0f);
            v=v+k*(theta-vpos)*dt+dW2*sqrtf(vpos)*sqrtf(dt)*xi;
            v=fmaxf(v,0.0f);
            S=S*expf((r-0.5f*v)*dt+sqrtf(v)*sqrtf(dt)*dW1);

        }
        payoffs[tid]=fmaxf(S-K,0.0f);
        state[tid]=localstate;
    }
}
__global__ void european_option_fwdgenp(float S,float *samples ,float *S_fwd_p, float K, float r,float s, float T,int nSim){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;

    if(tid<nSim){
        S_fwd_p[tid]=fmaxf(S*expf((r-0.5*s*s)*T + s*(sqrtf(T))*samples[tid])-K,0.0f);
    }
}

__global__ void sum_kernel(const float* __restrict__ in,float* __restrict__ partial, int n) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<n)
        partial[tid]=in[tid];
    else
        partial[tid]=0.0f;
}

__global__ void reduce_kernel(float* data, int n) {
    extern __shared__ float sdata[];

    int tid=threadIdx.x;
    float sum=0.0f;

    for(int i=tid;i<n;i+=blockDim.x){
        sum+=data[i];
    }
    sdata[tid]=sum;
    __syncthreads();

    for(int i=blockDim.x/2;i>0;i>>=1){
        if(tid<i){
            sdata[tid]+=sdata[tid+i];
        }
        __syncthreads();
    }
    if(tid==0)
        data[0]=sdata[0];
}



float european_option(float *arr,int n,float r,float T){
    float *d_arr, *d_partial;
    cudaMalloc(&d_arr,n*sizeof(float));
    cudaMalloc(&d_partial,n*sizeof(float));

    cudaMemcpy(d_arr,arr,n*sizeof(float),cudaMemcpyHostToDevice);

    int block=256;
    int grid=(n+block-1)/block;

    sum_kernel<<<grid,block>>>(d_arr,d_partial,n);
    reduce_kernel<<<1,block, block*sizeof(float)>>>(d_partial,n);
    float sum;
    cudaMemcpy(&sum, d_partial,sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_arr);cudaFree(d_partial);
    return std::exp(-r*T)*sum/n;
}

int main(){
    float S0    = 100.0f;
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

    european_option_heston<<<grid, block>>>(S0,d_payoffs,K,r,v0,T,k,theta,rho,xi,nSim,nSteps,d_state);
    cudaDeviceSynchronize();

    float *h_payoffs=new float[nSim];
    cudaMemcpy(h_payoffs,d_payoffs, nSim*sizeof(float),cudaMemcpyDeviceToHost);
    float price=european_option(h_payoffs,nSim,r,T);
    delete[] h_payoffs;
    cudaFree(d_state); cudaFree(d_payoffs);
    std::cout<<price<<"\n";
}