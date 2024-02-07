#!/usr/bin/env bash

#name the job nodeinfo and place it's output in a file named slurm-<jobid>.out
#set partition to 'all' this isn't strictly necessary but it's good practice
#set time to 5 minutes so jobs get killed if something weird happens

#SBATCH -J 'hmm_independent_biases_loocv'
#SBATCH -o ../logs/hmm_independent_biases_loocv-%j.out
#SBATCH -p all
#SBATCH -t 900
#SBATCH -c 8

echo $SLURM_ARRAY_TASK_ID

mkdir -p ../results/hmm_independent_biases_loocv
JULIA_NUM_THREADS=8 julia hmm_independent_biases_loocv.jl $SLURM_ARRAY_TASK_ID
