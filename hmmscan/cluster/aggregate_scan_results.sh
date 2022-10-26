#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=0-03:00:00
#SBATCH -o /home/jtwilde/projects/hmmscan/hmmscan/cluster/output.out
#SBATCH -e /home/jtwilde/projects/hmmscan/hmmscan/cluster/error.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jtwilde@mit.edu

module load R/4.1.0
Rscript hmmscan/scripts/scans/aggregate_scan_results.R $1