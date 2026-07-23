#pragma once

#include "../core/common.cuh"
#include "../core/types.cuh"
#include "../kernels/gemm_cublas.cuh"
#include "../kernels/abft_stepwise.cuh"
#include "../kernels/swifi.cuh"
#include "../metrics/metrics.cuh"
#include "buffers.cuh"


inline double pass_baseline(PipelineBuffers& b,
                            const float* dA, int lda,
                            const float* dB, int ldb,
                            float*       dC, int ldc,
                            int M_b, int K, int N_b,
                            int repeats) {
    (void)N_b;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto t0 = clk::now();

    for (int it = 0; it < repeats; ++it) {
        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            gemm_cublas(b.handle, dA, lda, dB + off, ldb, dC + off, ldc,
                        M_b, N_frag, K);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(b.compute_stream));
    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

inline double pass_calibrate(PipelineBuffers& b,
                             const float* dA, int lda,
                             const float* dB, int ldb,
                             float*       dC, int ldc,
                             int M_b, int K, int N_b,
                             double& out_max_diff,
                             std::vector<double>& diffs_out) {
    (void)N_b;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto t0 = clk::now();

    for (int f = 0; f < b.F; ++f) {
        int N_frag = b.col_counts[f];
        int off    = b.col_offsets[f];
        gemm_cublas(b.handle, dA, lda, dB + off, ldb, dC + off, ldc,
                    M_b, N_frag, K);
    }
    CUDA_CHECK(cudaStreamSynchronize(b.compute_stream));

    launch_col_checksum_A(dA, lda, b.dColSumA, M_b, K, b.verify_stream,
                          b.dEncPart);
    CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

    std::vector<double> hExp(b.N_frag_max), hAct(b.N_frag_max);
    for (int f = 0; f < b.F; ++f) {
        int N_frag = b.col_counts[f];
        int off    = b.col_offsets[f];
        launch_expected_row(b.dColSumA, dB + off, ldb,
                            b.dExpectedRow[f], K, N_frag, b.verify_stream,
                            b.dEncPart);
        launch_actual_row  (dC + off,   ldc,    b.dActualRow  [f],
                            M_b, N_frag, b.verify_stream);
        CUDA_CHECK(cudaMemcpyAsync(hExp.data(), b.dExpectedRow[f],
                                   sizeof(double) * N_frag,
                                   cudaMemcpyDeviceToHost, b.verify_stream));
        CUDA_CHECK(cudaMemcpyAsync(hAct.data(), b.dActualRow[f],
                                   sizeof(double) * N_frag,
                                   cudaMemcpyDeviceToHost, b.verify_stream));
        CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

        for (int j = 0; j < N_frag; ++j) {
            double d = std::abs(hAct[j] - hExp[j]);
            diffs_out.push_back(d);
            if (d > out_max_diff) out_max_diff = d;
        }
    }

    auto t1 = clk::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}


inline void pass_online_loop(PipelineBuffers& b,
                             const float* dA, int lda,
                             const float* dB, int ldb,
                             float* dC_buf0, float* dC_buf1, int ldc,
                             int M_b, int K, int N_b,
                             const std::vector<double>& thresholds,
                             const std::string& inject_mode,
                             const std::string& inject_zone,
                             uint64_t base_seed, int world_rank,
                             const std::vector<float>& C_golden,
                             int repeats,
                             std::vector<double>& out_iter_ms,
                             ConfusionMatrix& cm,
                             int& n_restored,
                             double& out_total_ms,
                             const std::string& encoding_mode = "amortized") {
    float* dC_bufs[2] = { dC_buf0, dC_buf1 };
    const bool inject_on  = (inject_mode == "swifi" || inject_mode == "add");
    const bool do_localize = inject_on;

    if (do_localize && !C_golden.empty() && b.dGolden == nullptr) {
        CUDA_CHECK(cudaMalloc(&b.dGolden,
                              sizeof(float) * (size_t)M_b * (size_t)N_b));
        CUDA_CHECK(cudaMemcpy(b.dGolden, C_golden.data(),
                              sizeof(float) * (size_t)M_b * (size_t)N_b,
                              cudaMemcpyHostToDevice));
    }

    cudaEvent_t buf_verify_done[2];
    bool buf_event_used[2] = { false, false };
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaEventCreateWithFlags(&buf_verify_done[i],
                                            cudaEventDisableTiming));

    auto enqueue_encode = [&]() {
        launch_col_checksum_A(dA, lda, b.dColSumA, M_b, K, b.verify_stream,
                              b.dEncPart);
        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            launch_expected_row(b.dColSumA, dB + off, ldb,
                                b.dExpectedRow[f], K, N_frag, b.verify_stream,
                                b.dEncPart);
        }
        CUDA_CHECK(cudaMemsetAsync(b.dCM, 0, sizeof(int) * 4, b.verify_stream));
        CUDA_CHECK(cudaMemsetAsync(b.dNRestored, 0, sizeof(int), b.verify_stream));
    };

    clk::time_point loop_t0;
    if (encoding_mode == "timed") {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        loop_t0 = clk::now();
        enqueue_encode();
        CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));
    } else if (encoding_mode == "overlap") {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        loop_t0 = clk::now();
        enqueue_encode();
    } else {
        enqueue_encode();
        CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        loop_t0 = clk::now();
    }

    for (int it = 0; it < repeats; ++it) {
        int buf_idx = it % 2;
        float* dC   = dC_bufs[buf_idx];

        if (buf_event_used[buf_idx]) {
            CUDA_CHECK(cudaStreamWaitEvent(b.compute_stream,
                                           buf_verify_done[buf_idx], 0));
        }

        int      inject_frag = -1;
        uint64_t inject_seed = base_seed
                             + (uint64_t)world_rank * 7919ull
                             + (uint64_t)it          * 104729ull;
        if (inject_on) {
            std::mt19937_64 rng(inject_seed);
            std::uniform_int_distribution<int> d(0, b.F - 1);
            inject_frag = d(rng);
        }

        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            gemm_cublas(b.handle, dA, lda, dB + off, ldb, dC + off, ldc,
                        M_b, N_frag, K);
            if (f == inject_frag) {
                if (inject_mode == "add")
                    inject_add_constant(dC + off, ldc, M_b, N_frag, f,
                                        inject_seed, b.compute_stream);
                else
                    inject_single_bitflip(dC + off, ldc, M_b, N_frag, f,
                                          inject_seed, b.compute_stream,
                                          inject_zone);
            }
            CUDA_CHECK(cudaEventRecord(b.compute_done[f], b.compute_stream));
        }

        for (int f = 0; f < b.F; ++f) {
            int N_frag = b.col_counts[f];
            int off    = b.col_offsets[f];
            int injected = (f == inject_frag) ? 1 : 0;

            CUDA_CHECK(cudaStreamWaitEvent(b.verify_stream,
                                           b.compute_done[f], 0));
            launch_actual_row(dC + off, ldc, b.dActualRow[f],
                              M_b, N_frag, b.verify_stream);
            launch_detect_row(b.dExpectedRow[f], b.dActualRow[f],
                              N_frag, thresholds[f],
                              b.dErrCol + f, b.dRowDiff + f,
                              injected, b.dCM, b.verify_stream);
            if (do_localize) {
                launch_localize_correct(dA, lda, dB + off, ldb,
                                        dC + off, ldc,
                                        b.dRowSumB, b.dExpectedCol,
                                        b.dActualCol,
                                        M_b, K, N_frag, thresholds[f],
                                        b.dErrCol + f, b.dRowDiff + f,
                                        b.dGolden, N_b, off, injected,
                                        b.dNRestored, b.verify_stream);
            }
        }

        CUDA_CHECK(cudaEventRecord(buf_verify_done[buf_idx], b.verify_stream));
        buf_event_used[buf_idx] = true;
    }

    CUDA_CHECK(cudaStreamSynchronize(b.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(b.verify_stream));

    auto loop_t1 = clk::now();
    out_total_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();

    int hCM[4] = {0, 0, 0, 0};
    int hNR    = 0;
    CUDA_CHECK(cudaMemcpy(hCM, b.dCM, sizeof(int) * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hNR, b.dNRestored, sizeof(int), cudaMemcpyDeviceToHost));
    cm.TP += hCM[0];
    cm.TN += hCM[1];
    cm.FP += hCM[2];
    cm.FN += hCM[3];
    n_restored += hNR;

    out_iter_ms.assign(repeats,
                       repeats > 0 ? out_total_ms / repeats : 0.0);

    for (int i = 0; i < 2; ++i) cudaEventDestroy(buf_verify_done[i]);
}
