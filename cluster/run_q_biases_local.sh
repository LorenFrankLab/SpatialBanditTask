#!/bin/bash
set -euo pipefail

mkdir -p ../results/q_biases
for i in {1..220}; do
    JULIA_NUM_THREADS=8 julia q_biases.jl $i
done 