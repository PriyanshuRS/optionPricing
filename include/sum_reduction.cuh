#pragma once

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