#!/bin/bash
set -euo pipefail

N_CONDS=15
N_CORES=8
MAX_TIME=600
STUDY_NAME="hmm_independentsamedist_staygo_biases"
VER_NAME=$STUDY_NAME

SCRIPT_NAME="slurm.${VER_NAME}.sh"
SLURM_RUN_NAME="run.slurm.${VER_NAME}.sh"
LOCAL_RUN_NAME="run.local.${VER_NAME}.sh"

mkdir -p ../results/${VER_NAME}

cat << EOF > slurm_scripts/$SCRIPT_NAME
#!/usr/bin/env bash

#SBATCH -J '${VER_NAME}'
#SBATCH -o ../logs/${VER_NAME}-%j.out
#SBATCH -p all
#SBATCH -t $MAX_TIME
#SBATCH -c $N_CORES

echo \$SLURM_ARRAY_TASK_ID

JULIA_NUM_THREADS=$N_CORES julia ${STUDY_NAME}.jl \$SLURM_ARRAY_TASK_ID false
EOF
chmod u+x slurm_scripts/$SCRIPT_NAME

cat << EOF > $SLURM_RUN_NAME
#!/bin/bash
sbatch --array=1-$N_CONDS slurm_scripts/$SCRIPT_NAME
EOF
chmod u+x $SLURM_RUN_NAME

cat << EOF > $LOCAL_RUN_NAME
#!/bin/bash

for i in {1..$N_CONDS}; do
    julia ${STUDY_NAME}.jl \$i false
done 
EOF
chmod u+x $LOCAL_RUN_NAME