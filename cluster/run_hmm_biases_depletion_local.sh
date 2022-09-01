#!/bin/bash
set -euo pipefail

mkdir -p ../results/hmm_biases_depletion
for i in {1..560}; do
    JULIA_NUM_THREADS=8 julia hmm_biases_depletion.jl $i
done 