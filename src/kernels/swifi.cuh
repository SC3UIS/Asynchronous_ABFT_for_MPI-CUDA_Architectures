#pragma once

#include "../core/common.cuh"
#include "../core/types.cuh"

inline void swifi_zone_range(const std::string& zone, int& lo, int& hi) {
    if      (zone == "sign")     { lo = 31; hi = 31; }
    else if (zone == "exponent") { lo = 23; hi = 30; }
    else if (zone == "sig_high") { lo = 13; hi = 22; }
    else if (zone == "sig_low")  { lo =  0; hi = 12; }
    else                         { lo =  0; hi = 31; }
}

__global__ inline void k_inject_bitflip(float* C, int ldc,
                                        int target_row, int target_col,
                                        int bit_position) {
    unsigned int* ptr = reinterpret_cast<unsigned int*>(&C[target_row * ldc + target_col]);
    unsigned int val = *ptr;
    val ^= (1u << bit_position);
    *ptr = val;
}

constexpr float ADD_FAULT_DELTA = 1.0e6f;

__global__ inline void k_inject_add(float* C, int ldc,
                                    int target_row, int target_col,
                                    float delta) {
    C[target_row * ldc + target_col] += delta;
}

inline InjectionInfo inject_add_constant(float* dC_frag, int ldc,
                                         int M_b, int N_frag, int frag_idx,
                                         uint64_t seed, cudaStream_t stream) {
    InjectionInfo info{};
    info.injected   = true;
    info.frag_index = frag_idx;
    std::mt19937_64 gen(seed + static_cast<uint64_t>(frag_idx));
    std::uniform_int_distribution<int> row_dist(0, M_b - 1);
    std::uniform_int_distribution<int> col_dist(0, N_frag - 1);
    info.row          = row_dist(gen);
    info.col          = col_dist(gen);
    info.bit_position = -1;
    k_inject_add<<<1, 1, 0, stream>>>(
        dC_frag, ldc, info.row, info.col, ADD_FAULT_DELTA);
    return info;
}

inline InjectionInfo inject_single_bitflip(float* dC_frag, int ldc,
                                           int M_b, int N_frag,
                                           int frag_idx,
                                           uint64_t seed,
                                           cudaStream_t stream,
                                           const std::string& zone = "any") {
    InjectionInfo info{};
    info.injected   = true;
    info.frag_index = frag_idx;

    int bit_lo, bit_hi;
    swifi_zone_range(zone, bit_lo, bit_hi);

    std::mt19937_64 gen(seed + static_cast<uint64_t>(frag_idx));
    std::uniform_int_distribution<int> row_dist(0, M_b - 1);
    std::uniform_int_distribution<int> col_dist(0, N_frag - 1);
    std::uniform_int_distribution<int> bit_dist(bit_lo, bit_hi);

    info.row          = row_dist(gen);
    info.col          = col_dist(gen);
    info.bit_position = bit_dist(gen);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&info.value_before,
                          &dC_frag[info.row * ldc + info.col],
                          sizeof(float), cudaMemcpyDeviceToHost));

    k_inject_bitflip<<<1, 1, 0, stream>>>(
        dC_frag, ldc, info.row, info.col, info.bit_position);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&info.value_after,
                          &dC_frag[info.row * ldc + info.col],
                          sizeof(float), cudaMemcpyDeviceToHost));

    return info;
}
