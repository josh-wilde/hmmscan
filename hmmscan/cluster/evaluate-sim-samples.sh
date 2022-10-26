#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100M
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-18:00:00
#SBATCH -o /home/jtwilde/projects/hmmscan/hmmscan/cluster/output.out
#SBATCH -e /home/jtwilde/projects/hmmscan/hmmscan/cluster/error.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jtwilde@mit.edu

module load python/3.9.4
python -m hmmscan.scripts.validation.evaluate-samples --validation_type simulation --exp_ids $SLURM_ARRAY_TASK_ID