#!/usr/bin/env bash

#name the job nodeinfo and place it's output in a file named slurm-<jobid>.out
#set partition to 'all' this isn't strictly necessary but it's good practice
#set time to 5 minutes so jobs get killed if something weird happens

#SBATCH -J 'q_biases'
#SBATCH -o ../logs/q_biases-%j.out
#SBATCH -p all
#SBATCH -t 300
#SBATCH -c 8

echo $SLURM_ARRAY_TASK_ID

module load julia/1.6.0

mkdir -p ../results/q_biases
JULIA_NUM_THREADS=8 julia q_biases.jl $SLURM_ARRAY_TASK_ID
