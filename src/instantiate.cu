#include "../include/options/options.cuh"
#include "../include/cpu/options.h"
#include "../include/options/payoffs.cuh"

template class European<Call>;
template class European<Put>;

template class Asian<Call>;
template class Asian<Put>;

template class Lookback<Call>;
template class Lookback<Put>;

template class CpuEuropean<Call>;
template class CpuEuropean<Put>;

template class CpuAsian<Call>;
template class CpuAsian<Put>;

template class CpuLookback<Call>;
template class CpuLookback<Put>;