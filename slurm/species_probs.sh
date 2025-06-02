#!/bin/bash

set -a
source .env
set +a

mkdir -p ./slurm/logs

sbatch --cpus-per-task=$NUM_WORKERS ./slurm/jobs/species_probs.job
