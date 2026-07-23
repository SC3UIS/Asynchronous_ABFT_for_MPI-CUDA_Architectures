#!/bin/bash
#SBATCH --job-name=abft_gemm
#SBATCH --partition=gpu_titan
#SBATCH --nodelist=felix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --output=scripts/abft_felix_%j.out
#SBATCH --error=scripts/abft_felix_%j.err

set -euo pipefail

module purge
module load cuda/11.8
module load mpi/openmpi-gcc11/4.1.6

MPICC_PATH="$(which mpicc)"
MPI_PREFIX="$(dirname "$(dirname "$MPICC_PATH")")"

echo "=== Compiling ABFT GEMM (cuBLAS, online pipelined) ==="
nvcc -O3 -std=c++17 -ccbin g++ \
  -I"${MPI_PREFIX}/include" \
  src/main.cu -o abft_gemm \
  -L"${MPI_PREFIX}/lib" -lmpi -lcublas
echo "Compilation OK"
echo

echo "=== GPU inventory ==="
nvidia-smi -L || true
echo

F=${F:-8}
R=${R:-1}
WARMUPS=${WARMUPS:-2}
SAFETY=${SAFETY:-10}
ENC_FLAG=(--encoding-mode overlap)

SWEEP_N=${SWEEP_N:-40}
SWEEP_SAMPLES=${SWEEP_SAMPLES:-20}
sweep_grid() { local i; for ((i=0;i<SWEEP_N;i++)); do echo $(( 256*(i+1) )); done; }
mapfile -t SWEEP_SIZES < <( sweep_grid | awk '!s[$0]++' )

SWEEP_K_NS=${SWEEP_K_NS:-1024}

CM_SIZES=( ${CM_SIZES:-2048 4096 8192} )
CM_SAMPLES=${CM_SAMPLES:-100}
ZONES=( ${ZONES:-any sign exponent sig_high sig_low} )

TAU=${TAU:-calibrate}

CALIB_R=${CALIB_R:-20}

rm -f abft_metrics.csv abft_metrics_ns.csv

run() {
  local label="$1"; shift
  echo "============================================"
  echo "=== ${label} ==="
  echo "============================================"
  srun --mpi=pmix ./abft_gemm "$@"
  echo
}

calibrate_tau() {
  local S="$1" cf="_calib_${S}.csv"
  rm -f "${cf}"
  srun --mpi=pmix ./abft_gemm "${S}" "${S}" "${S}" --calibrate \
       --frags-per-rank ${F} --repeats ${CALIB_R} \
       --calibration-safety ${SAFETY} --csv "${cf}" >/dev/null 2>&1 || true
  python3 - "${cf}" <<'PY'
import csv,sys
tau=""
try:
    for row in csv.DictReader(open(sys.argv[1])): tau=row.get("calib_suggested_tau","")
except Exception: pass
print(tau or "")
PY
}

echo "############################################"
echo "###  PERF SWEEP  (${#SWEEP_SIZES[@]} sizes, ${SWEEP_SAMPLES} samples)"
echo "############################################"

SWEEP_COMMON=(--frags-per-rank ${F} --samples ${SWEEP_SAMPLES}
              --repeats ${R} --warmups ${WARMUPS})
SWEEP_TAU="1.0"

for S in "${SWEEP_SIZES[@]}"; do
  echo "###  sweep ${S}^3"
  run "SWEEP baseline ${S}^3" ${S} ${S} ${S} --baseline                     "${SWEEP_COMMON[@]}"
  run "SWEEP online  ${S}^3"  ${S} ${S} ${S} --inject none --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON[@]}"
  run "SWEEP online+add ${S}^3" ${S} ${S} ${S} --inject add  --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON[@]}"
done

echo "############################################"
echo "###  PERF SWEEP NS  (${#SWEEP_SIZES[@]} sizes, K=${SWEEP_K_NS} fixed, ${SWEEP_SAMPLES} samples)"
echo "############################################"

SWEEP_COMMON_NS=("${SWEEP_COMMON[@]}" --csv abft_metrics_ns.csv)

for S in "${SWEEP_SIZES[@]}"; do
  echo "###  sweep_ns ${S}x${SWEEP_K_NS}x${S}"
  run "SWEEP_NS baseline ${S}x${SWEEP_K_NS}x${S}"   ${S} ${SWEEP_K_NS} ${S} --baseline                     "${SWEEP_COMMON_NS[@]}"
  run "SWEEP_NS online   ${S}x${SWEEP_K_NS}x${S}"   ${S} ${SWEEP_K_NS} ${S} --inject none --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON_NS[@]}"
  run "SWEEP_NS online+add ${S}x${SWEEP_K_NS}x${S}" ${S} ${SWEEP_K_NS} ${S} --inject add  --threshold ${SWEEP_TAU} "${ENC_FLAG[@]}" "${SWEEP_COMMON_NS[@]}"
done

echo "############################################"
echo "###  CM CAMPAIGN  (${#CM_SIZES[@]} sizes, ${CM_SAMPLES} samples × ${#ZONES[@]} zones)"
echo "############################################"

CM_COMMON=(--frags-per-rank ${F} --samples ${CM_SAMPLES}
           --reseed-per-trial --repeats ${R} --warmups ${WARMUPS})

for S in "${CM_SIZES[@]}"; do
  case "${TAU}" in
    formula)
      TH=(); tau="<formula>" ;;
    calibrate|"")
      echo "### Calibrating tau for ${S}^3 ..."
      tau="$(calibrate_tau "${S}")"
      if [[ -z "${tau}" || "${tau}" == "0.000000" ]]; then
        echo "  WARNING: calibration gave no tau for ${S}^3; using formula"
        TH=(); tau="<formula>"
      else
        TH=(--threshold "${tau}")
      fi ;;
    *)
      tau="${TAU}"; TH=(--threshold "${TAU}") ;;
  esac
  echo "### CM ${S}^3  tau=${tau}"

  for z in "${ZONES[@]}"; do
    run "CM online/swifi[${z}] ${S}^3" ${S} ${S} ${S} \
        --inject swifi --swifi-zone "${z}" "${TH[@]}" "${ENC_FLAG[@]}" "${CM_COMMON[@]}"
  done
done

echo "=== All runs complete ==="
echo "CSV: $(pwd)/abft_metrics.csv"

echo
echo "=== Rendering plots ==="
mkdir -p scripts/plots/campaign scripts/plots/campaign_ns
python3 scripts/plot_metrics.py \
        --metrics abft_metrics.csv \
        --out     scripts/plots/campaign \
    || echo "(skipped: install pandas matplotlib numpy on the cluster)"
if [[ -s abft_metrics_ns.csv ]]; then
    python3 scripts/plot_metrics.py \
            --metrics abft_metrics_ns.csv \
            --out     scripts/plots/campaign_ns \
        || echo "(ns plots skipped)"
fi
