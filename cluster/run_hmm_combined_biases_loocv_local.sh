#!/bin/bash
set -euo pipefail

mkdir -p ../results/hmm_independent_combined_biases_loocv
for i in {1..160}; do
    JULIA_NUM_THREADS=8 julia hmm_independent_combined_biases_loocv.jl $i
done 