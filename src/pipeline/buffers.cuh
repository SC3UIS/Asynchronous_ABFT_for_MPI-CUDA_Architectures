#pragma once

#include "../core/common.cuh"
#include "../distribution/grid.cuh"
#include "../kernels/abft_stepwise.cuh"

struct PipelineBuffers {
    int F;
    int N_frag_max;
    int M_b;
    int N_b;
    int K;
    int R;

    std::vector<double*> dExpectedRow;
    std::vector<double*> dActualRow;
    double*              dColSumA = nullptr;

    int*    dErrCol   = nullptr;
    double* dRowDiff  = nullptr;

    double* dRowSumB     = nullptr;
    double* dExpectedCol = nullptr;
    double* dActualCol   = nullptr;

    double* dEncPart     = nullptr;

    int* dCM        = nullptr;
    int* dNRestored = nullptr;

    float* dGolden = nullptr;

    std::vector<cudaEvent_t> compute_done;

    cudaStream_t   compute_stream;
    cudaStream_t   verify_stream;
    cublasHandle_t handle;

    std::vector<int> col_counts;
    std::vector<int> col_offsets;
};

inline void buffers_init(PipelineBuffers& b, int F, int M_b, int N_b, int K, int R) {
    b.F   = F;
    b.M_b = M_b;
    b.N_b = N_b;
    b.K   = K;
    b.R   = R;

    split_dim(N_b, F, b.col_counts, b.col_offsets);
    b.N_frag_max = *std::max_element(b.col_counts.begin(), b.col_counts.end());

    b.dExpectedRow.resize(F);
    b.dActualRow  .resize(F);
    b.compute_done.resize(F);

    for (int f = 0; f < F; ++f) {
        CUDA_CHECK(cudaMalloc(&b.dExpectedRow[f], sizeof(double) * b.N_frag_max));
        CUDA_CHECK(cudaMalloc(&b.dActualRow  [f], sizeof(double) * b.N_frag_max));
        CUDA_CHECK(cudaEventCreateWithFlags(&b.compute_done[f], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaMalloc(&b.dColSumA, sizeof(double) * K));

    CUDA_CHECK(cudaMalloc(&b.dErrCol,      sizeof(int)    * F));
    CUDA_CHECK(cudaMalloc(&b.dRowDiff,     sizeof(double) * F));
    CUDA_CHECK(cudaMalloc(&b.dRowSumB,     sizeof(double) * K));
    CUDA_CHECK(cudaMalloc(&b.dExpectedCol, sizeof(double) * M_b));
    CUDA_CHECK(cudaMalloc(&b.dActualCol,   sizeof(double) * M_b));
    CUDA_CHECK(cudaMalloc(&b.dCM,          sizeof(int)    * 4));
    CUDA_CHECK(cudaMalloc(&b.dNRestored,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b.dEncPart,
                          sizeof(double) * (size_t)ENC_CHUNKS * (K > N_b ? K : N_b)));
    b.dGolden = nullptr;

    CUDA_CHECK(cudaStreamCreate(&b.compute_stream));
    CUDA_CHECK(cudaStreamCreate(&b.verify_stream));

    CUBLAS_CHECK(cublasCreate(&b.handle));
    CUBLAS_CHECK(cublasSetStream(b.handle, b.compute_stream));
}

inline void buffers_free(PipelineBuffers& b) {
    for (int f = 0; f < b.F; ++f) {
        cudaFree(b.dExpectedRow[f]);
        cudaFree(b.dActualRow[f]);
        cudaEventDestroy(b.compute_done[f]);
    }
    cudaFree(b.dColSumA);
    cudaFree(b.dErrCol);
    cudaFree(b.dRowDiff);
    cudaFree(b.dRowSumB);
    cudaFree(b.dExpectedCol);
    cudaFree(b.dActualCol);
    cudaFree(b.dCM);
    cudaFree(b.dNRestored);
    cudaFree(b.dEncPart);
    if (b.dGolden) cudaFree(b.dGolden);

    cudaStreamDestroy(b.compute_stream);
    cudaStreamDestroy(b.verify_stream);
    cublasDestroy(b.handle);
}
