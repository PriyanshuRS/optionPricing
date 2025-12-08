#pragma once
#include<cmath>

class Call{
    private:
        float S,K;
    public:
        __device__ __host__ float operator()(float S,float K){
            return fmaxf(S-K,0.0f);
        }
};
class Put{
    private:
        float S,K;
    public:
        __device__ __host__ float operator()(float S,float K){
            return fmaxf(K-S,0.0f);
        }
};