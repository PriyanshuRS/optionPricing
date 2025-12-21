#pragma once
#include <iostream>
#include <sstream>
#include <string>

enum class OptionType{
    European=1,
    Asian=2,
    Lookback=3
};

enum class PayoffType{
    Call=1,
    Put=2
};

struct InputParams{
    bool runCpu;
    OptionType option;
    PayoffType payoff;

    float S0;
    float K;
    float r;
    float T_days;
    float v0;
    float k;
    float theta;
    float rho;
    float xi;

    int nSim;
    int nSteps;
};


template <typename T>
T prompt_value(const std::string& label,T default_val){
    std::cout<<label<<" ["<<default_val<<"]: ";

    std::string line;
    std::getline(std::cin,line);

    if (line.empty())
        return default_val;

    std::stringstream ss(line);
    T value;
    ss>>value;
    return value;
}

static bool prompt_run_cpu(){
    std::cout << "Run CPU simulation for comparison? (1/0) [0]: ";
    std::string line;
    std::getline(std::cin, line);

    if (line.empty()) return false;

    int choice=std::stoi(line);
    return choice;
}

static OptionType prompt_option_type(){
    std::cout<< "\nSelect option type:\n";
    std::cout<< "  1) European\n";
    std::cout<< "  2) Asian\n";
    std::cout<< "  3) Lookback\n";
    std::cout<< "Choice [1]: ";

    std::string line;
    std::getline(std::cin,line);

    if (line.empty()) return OptionType::European;

    int choice=std::stoi(line);
    return static_cast<OptionType>(choice);
}

static PayoffType prompt_payoff_type(){
    std::cout<< "\nSelect payoff type:\n";
    std::cout<< "  1) Call\n";
    std::cout<< "  2) Put\n";
    std::cout<< "Choice [1]: ";

    std::string line;
    std::getline(std::cin,line);

    if (line.empty()) return PayoffType::Call;

    int choice=std::stoi(line);
    return static_cast<PayoffType>(choice);
}

InputParams input(){
    InputParams p{};

    std::cout<<"==== Monte Carlo Heston Model Option Pricing ====\n";

    p.runCpu = prompt_run_cpu();
    p.option =prompt_option_type();
    p.payoff =prompt_payoff_type();

    std::cout << "\nModel parameters:\n";

    p.S0=prompt_value("Spot price S0", 25966.4f);
    p.K =prompt_value("Strike K", 26000.0f);
    p.T_days=prompt_value("Maturity (days)", 2.0f);
    p.r =prompt_value("Risk-free rate r", 0.0738f);
    p.v0 = prompt_value("Initial Variance", 0.01f);
    p.k = prompt_value("Mean Reversion Rate", 0.0f);
    p.theta= prompt_value("Terminal Variance", 0.0f);
    p.rho =prompt_value("Correlation factor", 0.0f);
    p.xi=prompt_value("Volitility of Variance", 0.0f);

    std::cout<<"\nMonte Carlo parameters:\n";

    p.nSim = prompt_value("Number of simulations",1000000);
    p.nSteps = prompt_value("Time steps",1000);

    return p;
}

