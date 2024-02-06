#!/bin/bash
set -euo pipefail

mkdir -p ../results/hmm_independent_biases
for i in {1..190}; do
    JULIA_NUM_THREADS=8 julia hmm_independent_biases.jl $i
done 