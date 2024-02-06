#!/usr/bin/env bash

#name the job nodeinfo and place it's output in a file named slurm-<jobid>.out
#set partition to 'all' this isn't strictly necessary but it's good practice
#set time to 5 minutes so jobs get killed if something weird happens

#SBATCH -J 'hmm_independent_biases'
#SBATCH -o ../logs/hmm_independent_biases-%j.out
#SBATCH -p all
#SBATCH -t 300
#SBATCH -c 8

echo $SLURM_ARRAY_TASK_ID

mkdir -p ../results/hmm_independent_biases
JULIA_NUM_THREADS=8 julia hmm_independent_biases.jl $SLURM_ARRAY_TASK_ID
