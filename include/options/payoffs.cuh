#pragma once
#include<cmath>

class Call{
    public:
        __device__ __host__ float operator()(float S,float K){
            return fmaxf(S-K,0.0f);
        }
};
class Put{
    public:
        __device__ __host__ float operator()(float S,float K){
            return fmaxf(K-S,0.0f);
        }
};