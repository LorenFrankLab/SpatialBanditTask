#!/usr/bin/env bash
#this script dumps some basic information to an output file when submitted via sbatch

#name the job nodeinfo and place it's output in a file named slurm-<jobid>.out
#set partition to 'all' this isn't strictly necessary but it's good practice
#set time to 5 minutes so jobs get killed if something weird happens

#SBATCH -J 'hmm_biases_depletion'
#SBATCH -o ../logs/hmm_biases_depletion-%j.out
#SBATCH -p all
#SBATCH -t 300
#SBATCH -c 8

echo $SLURM_ARRAY_TASK_ID

mkdir -p ../results/hmm_biases_depletion
JULIA_NUM_THREADS=8 julia hmm_biases_depletion.jl $SLURM_ARRAY_TASK_ID
