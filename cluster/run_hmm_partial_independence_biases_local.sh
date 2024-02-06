#!/bin/bash
set -euo pipefail

mkdir -p ../results/hmm_partial_independence_biases
for i in {1..960}; do
    JULIA_NUM_THREADS=8 julia hmm_partial_independence_biases.jl $i
done 