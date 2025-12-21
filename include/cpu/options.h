#pragma once

#include<iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>

#include "../options/payoffs.cuh"

template <typename CallPut>
class CpuEuropean{
private:
    float S0,K,T,r,v0,k,theta,rho,xi;
public:
    CpuEuropean(float _S0,float _K,float _T,float _r,float _v0,float _k,float _theta,float _rho,float _xi) : S0(_S0),K(_K),T(_T),r(_r),v0(_v0),k(_k),theta(_theta),rho(_rho),xi(_xi){}

    void simulate(std::vector<float>& payoffs, unsigned int nSim,unsigned int nSteps){
        std::mt19937 generator(1234);
        std::normal_distribution<float> distribution(0.0,1.0);
        
        float dt=T/(float)nSteps;
        CallPut payoff;

        for(unsigned int i=0;i<nSim;i++){
            float S=S0;
            float v=v0;

            for(unsigned int j=0;j<nSteps;j++){
                float dW1=distribution(generator);
                float dW2= rho*dW1+sqrtf(1.0f-rho*rho)*distribution(generator);
                float vpos=fmaxf(v,0.0f);
                S=S*expf((r-0.5f*vpos)*dt + sqrtf(vpos)*sqrtf(dt)*dW1);
                v=v+k*(theta-vpos)*dt+xi*sqrtf(vpos)*dW2*sqrtf(dt);
                v=fmaxf(v,0.0f);
            }
            payoffs[i]=payoff(S,K);
        }
    }

    float calculate(const std::vector<float>& payoffs,float r,float T){
        float sum=std::accumulate(payoffs.begin(),payoffs.end(),0.0f);
        return std::exp(-r*T)*(sum/payoffs.size());
    }
};

template <typename CallPut>
class CpuAsian{
private:
    float S0,K,T,r,v0,k,theta,rho,xi;

public:
    CpuAsian(float _S0,float _K,float _T,float _r,float _v0,float _k,float _theta,float _rho,float _xi) : S0(_S0),K(_K),T(_T),r(_r),v0(_v0),k(_k),theta(_theta),rho(_rho),xi(_xi){}

    void simulate(std::vector<float>& payoffs, unsigned int nSim,unsigned int nSteps){
        std::mt19937 generator(1234);
        std::normal_distribution<float> distribution(0.0,1.0);
        
        float dt=T/(float)nSteps;
        CallPut payoff;

        for(unsigned int i=0;i<nSim;i++){
            float S=S0;
            float v=v0;
            double sum=S0;

            for(unsigned int j=0;j<nSteps;j++){
                float dW1=distribution(generator);
                float dW2= rho*dW1+sqrtf(1.0f-rho*rho)*distribution(generator);
                float vpos=fmaxf(v,0.0f);
                S=S*expf((r-0.5f*vpos)*dt + sqrtf(vpos)*sqrtf(dt)*dW1);
                v=v+k*(theta-vpos)*dt+xi*sqrtf(vpos)*dW2*sqrtf(dt);
                v=fmaxf(v,0.0f);
                sum+=S;
            }
            float S_avg=sum/(nSteps+1);
            payoffs[i]=payoff(S_avg,K);
        }
    }

    float calculate(const std::vector<float>& payoffs,float r,float T){
        float sum=std::accumulate(payoffs.begin(),payoffs.end(),0.0f);
        return std::exp(-r*T)*(sum/payoffs.size());
    }
};

template <typename CallPut>
class CpuLookback{
private:
    float S0,K,T,r,v0,k,theta,rho,xi;

public:
    CpuLookback(float _S0,float _K,float _T,float _r,float _v0,float _k,float _theta,float _rho,float _xi) : S0(_S0),K(_K),T(_T),r(_r),v0(_v0),k(_k),theta(_theta),rho(_rho),xi(_xi){}

    void simulate(std::vector<float>& payoffs, unsigned int nSim,unsigned int nSteps){
        std::mt19937 generator(1234);
        std::normal_distribution<float> distribution(0.0,1.0);
        
        float dt=T/(float)nSteps;
        CallPut payoff;

        for(unsigned int i=0;i<nSim;i++){
            float S=S0;
            float v=v0;
            float minn=S0;
            float maxx=S0;

            for(unsigned int j=0;j<nSteps;j++){
                float dW1=distribution(generator);
                float dW2= rho*dW1+sqrtf(1.0f-rho*rho)*distribution(generator);
                float vpos=fmaxf(v,0.0f);
                S=S*expf((r-0.5f*vpos)*dt + sqrtf(vpos)*sqrtf(dt)*dW1);
                v=v+k*(theta-vpos)*dt+xi*sqrtf(vpos)*dW2*sqrtf(dt);
                v=fmaxf(v,0.0f);
                minn=min(minn,S);
                maxx=max(maxx,S);
            }
            if constexpr(std::is_same_v<CallPut,Call>){
                payoffs[i]=payoff(maxx,K);
            }
            else{
                payoffs[i]=payoff(minn,K);
            }
        }
    }

    float calculate(const std::vector<float>& payoffs,float r,float T){
        float sum=std::accumulate(payoffs.begin(),payoffs.end(),0.0f);
        return std::exp(-r*T)*(sum/payoffs.size());
    }
};
