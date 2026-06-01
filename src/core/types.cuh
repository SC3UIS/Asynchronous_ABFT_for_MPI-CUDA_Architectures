#pragma once

#include "common.cuh"

struct Grid2D {
    int Pr;
    int Pc;
    int pr;
    int pc;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
};

struct ABFTResult {
    bool  error_detected   = false;
    bool  error_corrected  = false;
    int   frag_index       = -1;
    int   error_row        = -1;
    int   error_col        = -1;
    float corrupted_value  = 0.0f;
    float corrected_value  = 0.0f;
    float golden_value     = 0.0f;
};

struct ExperimentConfig {
    int         M = 1024;
    int         K = 1024;
    int         N = 1024;
    int         Pr = 0, Pc = 0;

    int         frags_per_rank = 4;
    int         repeats        = 20;

    int         warmups        = 2;

    int         frag_cap       = 0;

    int         num_samples    = 0;

    bool        reseed_per_trial = false;

    std::string scheme         = "online";
    std::string inject         = "none";

    std::string swifi_zone     = "any";
    bool        baseline_only  = false;
    bool        calibrate      = false;

    double      threshold_override = -1.0;

    double      calibration_safety_factor = 10.0;

    uint64_t    seed_a = 123456;
    uint64_t    seed_b = 987654;

    std::string csv_path        = "abft_metrics.csv";

    std::string calib_diffs_path = "abft_calibration_diffs.csv";
};

struct InjectionInfo {
    bool  injected     = false;
    int   frag_index   = -1;
    int   row          = -1;
    int   col          = -1;
    int   bit_position = -1;
    float value_before = 0.0f;
    float value_after  = 0.0f;
};
