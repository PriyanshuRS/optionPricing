#include "../include/input/input.cuh"
#include "run.cuh"

int main() {
    InputParams in=input();

    float T=in.T_days/252.0f;

    if (in.option == OptionType::European){
        if(in.payoff == PayoffType::Call){
            run_gpu<European<Call>>(in.S0,in.K,T,in.r,in.v0,in.k,in.theta,in.rho,in.xi,in.nSim,in.nSteps);
        }
        else{
            run_gpu<European<Put>>(in.S0,in.K,T,in.r,in.v0,in.k,in.theta,in.rho,in.xi,in.nSim,in.nSteps);
        }
    }
    else if(in.option==OptionType::Asian){
        if(in.payoff == PayoffType::Call){
            run_gpu<Asian<Call>>(in.S0,in.K,T,in.r,in.v0,in.k,in.theta,in.rho,in.xi,in.nSim,in.nSteps);
        }
        else{
            run_gpu<Asian<Put>>(in.S0,in.K,T,in.r,in.v0,in.k,in.theta,in.rho,in.xi,in.nSim,in.nSteps);
        }
    }
    else{
        if(in.payoff==PayoffType::Call){
            run_gpu<Lookback<Call>>(in.S0,in.K,T,in.r,in.v0,in.k,in.theta,in.rho,in.xi,in.nSim,in.nSteps);
        }
        else{
            run_gpu<Lookback<Put>>(in.S0,in.K,T,in.r,in.v0,in.k,in.theta,in.rho,in.xi,in.nSim,in.nSteps);
        }
    }

    return 0;
}