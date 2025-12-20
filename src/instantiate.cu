#include "../include/options/options.cuh"
#include "../include/options/payoffs.cuh"

template class European<Call>;
template class European<Put>;

template class Asian<Call>;
template class Asian<Put>;

template class Lookback<Call>;
template class Lookback<Put>;