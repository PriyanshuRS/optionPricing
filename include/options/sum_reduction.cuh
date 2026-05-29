#pragma once

static __global__ void reduce_blocks_kernel(const float* __restrict__ in, float* __restrict__ out, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum=0.0f;
    while(i<n){
        sum+=in[i];
        i+=blockDim.x*gridDim.x;
    }
    sdata[tid]=sum;
    __syncthreads();
    
    for(int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            sdata[tid]+=sdata[tid + s];
        }
        __syncthreads();
    }    
    if(tid==0) out[blockIdx.x]=sdata[0];
}

static __global__ void reduce_final_kernel(const float* __restrict__ in, float* __restrict__ out, int n){
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    float sum=0.0f;
    if(tid<n){
        sum=in[tid];
    }
    sdata[tid]=sum;
    __syncthreads();
    
    for(int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    if(tid==0) out[0]=sdata[0];
}