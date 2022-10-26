#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=2-00:00:00
#SBATCH -o /home/jtwilde/projects/hmmscan/hmmscan/cluster/output.out
#SBATCH -e /home/jtwilde/projects/hmmscan/hmmscan/cluster/error.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jtwilde@mit.edu

module load python/3.9.4

# Run the job
# array should be set to --array=0-99, one for each sample, during sbatch call.
for i in {0..35}
do
  if [ $i != $6 ]
  then
    python -m hmmscan.scripts.validation.evaluate-samples --validation_type use_case --init_file_path $1 --sequence_name $2 --ae_type $3 --eval_n_states $4 --eval_n_mix_comps $5 --grid_index $i --sample_index $SLURM_ARRAY_TASK_ID
  fi
done