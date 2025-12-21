#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

class CudaBenchmark{
private:
    cudaEvent_t start_,stop_;
    float elapsed_;
public:
    CudaBenchmark():elapsed_(0.0f){
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaBenchmark(){
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    inline void start(){
        cudaEventRecord(start_);
    }

    inline void stop(){
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed_, start_, stop_);
    }

    inline void report(const char* label,int nSim) const{
        double paths_per_sec=nSim/(elapsed_*1e-3);
        std::cout << label << ":\n"
                  << "  Time       : " << elapsed_ << " ms\n"
                  << "  Throughput : "
                  << paths_per_sec / 1e6
                  << " M paths/sec\n";
    }


};

class Timer {
private:
    std::chrono::steady_clock::time_point t0_,t1_;
public:
    void start(){
        t0_=std::chrono::steady_clock::now();
    }

    void stop(){
        t1_ =std::chrono::steady_clock::now();
    }

    void report() const{
        std::cout<<"Total time "<<std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(t1_-t0_).count()<<" ms\n";
    }

};
