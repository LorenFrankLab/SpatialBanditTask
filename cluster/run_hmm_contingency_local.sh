#!/bin/bash
set -euo pipefail

mkdir -p ../results/hmm_contingency
for i in {1..50}; do
    JULIA_NUM_THREADS=8 julia hmm_contingency.jl $i
done 