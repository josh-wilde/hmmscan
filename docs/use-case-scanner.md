# Use Case Scanner Documentation

This file contains instructions for Parameter Estimation when applying the HMMScan method to a use case dataset.
The instructions can be used either to replicate the use case in the paper results or to run HMMScan on a new sequence.

## Required Inputs
1. `sequence_name`: string, name of lot sequence (e.g., `dfa_by_date_ex_iqr_outliers`).
2. `ae_type`: string, name of AE type (e.g., `serious_std`). This is either `serious_std` or `exp_no_admin_std` for paper result replication.
3. `s_max`: integer, maximum number of candidate HMM states. This is 4 for paper result replication.
4. `c_max`: integer, maximum number of candidate HMM mixture components. This is 9 for paper result replication.
5. `output_subdir`: string, name of subdirectory of `ae-project/results/use_case/random_initializations` to store the HMM fitting results.

## 1. Input Data Setup
Ensure that the file `ae-project/data/use_case/[sequence_name].csv` has been created. If providing your own sequence, not replicating the paper results, then see the [User-Provided Input Data](new-use-case-data.md) for details.

### Paper Result Replication
Ensure that the [`ae-project` repository](https://doi.org/10.17632/zzd5vbj7yn.1) has been downloaded locally (see the repo readme file [here](../README.md) for instructions).
If this repository is downloaded, then the necessary input data files will already be available. 

## 2. Run the scanner on Engaging cluster
From the top level of this directory on Engaging, run the following command for each combination of `sequence_name` and `ae_type` to fit 1 state models:

`sbatch --array=0-49 --time=0-00:30:00 hmmscan/cluster/scan_use_case_parallel.sh seed_starter [sequence_name] [ae_type] 1 [output_subdir]`

Then, run the following command for each combination of `sequence_name` and `ae_type` for each state from 2 to `s_max`.

`sbatch --array=0-[c_max * 50 - 1] --time=0-01:00:00 hmmscan/cluster/scan_use_case_parallel.sh grid_component_seed_starter [sequence_name] [ae_type] [n_states] [output_subdir]`

### Paper Result Replication
The commands above must be run for each of the following `sequence_name`, `ae_type` combinations:
1. `dfa_by_date_ex_iqr_outliers`, `serious_std`
2. `dfb_by_date_ex_iqr_outliers`, `serious_std`
3. `dfc_by_date_ex_iqr_outliers`, `serious_std`
4. `dfa_by_date_ex_iqr_expedited`, `exp_no_admin_std`
5. `dfb_by_date_ex_iqr_expedited`, `exp_no_admin_std`
6. `dfc_by_date_ex_iqr_expedited`, `exp_no_admin_std`

Here is the command for the single state model fitting:

`sbatch --array=0-49 --time=0-00:30:00 hmmscan/cluster/scan_use_case_parallel.sh seed_starter [sequence_name] [ae_type] 1 by_date_ex_iqr`

Here is the command for the multiple state model fitting. This is run for `n_states` equal to 2, 3, then 4.

`sbatch --array=0-449 --time=0-01:00:00 hmmscan/cluster/scan_use_case_parallel.sh grid_component_seed_starter [sequence_name] [ae_type] [n_states] by_date_ex_iqr`

Step 2 generates a file for each combination of `sequence_name`, `ae_type`, number of hidden states, number of mixture components, and random initialization in `ae-project/results/use_case/random_initializations/[output_subdir]`.

## 3. Aggregate the parameter estimation information
On Engaging, run the following commands in an interactive session:

1. Load `R 4.1`: `module load R/4.1.0`.
2. Run `aggregate_scan_results.R`: `Rscript hmmscan/scripts/scans/aggregate_scan_results.R scans/use_case/random_initializations/[output_subdir]`

This script will generate a CSV file called `ae-project/results/scans/use_case/random_initializations/[output_subdir].csv`.

### Paper Result Replication
Use this command: `Rscript hmmscan/scripts/scans/aggregate_scan_results.R scans/use_case/random_initializations/by_date_ex_iqr`

## 4. Choose the best random initialization for each model structure
On Engaging, run the following commands in an interactive session:

1. Load `R 4.1`: `module load R/4.1.0`.
2. Run `get_best_initializations.R`: `Rscript hmmscan/scripts/scans/get_best_initializations.R [output_subdir].csv`

This script will generate a CSV file called `ae-project/results/scans/use_case/best_initializations/[output_subdir].csv`.

### Paper Result Replication
Use this command: `Rscript hmmscan/scripts/scans/get_best_initializations.R by_date_ex_iqr.csv`

## 5. Run state prediction
For this section, you will need to look at the CSV file generated in step 4 and find the best number of states and mixture components for each `sequence_name` and `ae_type` combination.
The best structure is referred to below as `best_n_states` and `best_n_mix_comps`.

On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `scripts/state_prediction.py`: `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/[output_subdir].csv use_case [sequence_name] [ae_type] [best_n_states] [best_n_mix_comps]`.

This script will generate a file in `ae-project/results/state_prediction/use_case` for each `sequence_name`, `ae_type`, `best_n_states`, and `best_n_mix_comps`.

### Paper Result Replication
Run these commands: 
1. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr.csv use_case dfa_by_date_ex_iqr_expedited exp_no_admin_std 3 3`
2. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr.csv use_case dfb_by_date_ex_iqr_expedited exp_no_admin_std 2 2`
3. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr.csv use_case dfc_by_date_ex_iqr_expedited exp_no_admin_std 1 3`
4. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr.csv use_case dfa_by_date_ex_iqr_outliers serious_std 3 2`
5. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr.csv use_case dfb_by_date_ex_iqr_outliers serious_std 3 3`
6. `python -m hmmscan.scripts.state_prediction.state-prediction scans/use_case/best_initializations/by_date_ex_iqr.csv use_case dfc_by_date_ex_iqr_outliers serious_std 2 3`

## 6. Visualize the results
It is probably easiest to generate the necessary plots locally off Engaging. To do so, copy `ae-project/results/scans/use_case/best_initializations/[output_subdir].csv` and the contents of `ae-project/results/state_prediction/use_case` into the same relative file locations in your local version of `ae-project`.

Then, you can run `hmmscan/scripts/viz/bic.R` to view the BICs of the HMM model candidates, and `hmmscan/scripts/viz/best_model_dists_and_predictions.R` to view the characteristics of the models with the best BICs.

If you are using your own lot sequence and not replicating the paper results, then you will need to adjust these visualization scripts.
