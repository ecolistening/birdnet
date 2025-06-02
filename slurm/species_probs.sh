#!/bin/bash

set -a
source .env
set +a

sbatch --cpus-per-task=$NUM_WORKERS ./slurm/jobs/species_probs.job

