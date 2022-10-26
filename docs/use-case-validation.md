# Use Case Validation Documentation

This file contains instructions for creating the Use Case Validation results.

## Required Inputs
1. `output_subdir`: string, name of file in `ae-project/results/scans/use_case/best_initializations` where the best initializations are stored.
2. `ae-project/results/scans/use_case/best_initializations/[output_subdir].csv`, generated by the use case scanner. For details, see [Use Case Scanner Documentation](use-case-scanner.md).

## 1. Generate samples
On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `python -m hmmscan.scripts.validation.generate-samples --validation_type use_case --init_file_name scans/use_case/best_initializations/[output_subdir].csv --sequence_name [sequence_name] --ae_type [ae_type] --n_samples 100`.

This generates a file for each `sequence_name`, `ae_type`, and model structure in `ae-project/validation/use_case/samples` and `ae-project/validation/use_case/state_paths`.

### Paper Result Replication
Run the following three commands:
1. `python -m hmmscan.scripts.validation.generate-samples --validation_type use_case --init_file_name scans/use_case/best_initializations/by_date_ex_iqr.csv --sequence_name dfa_by_date_ex_iqr_outliers --ae_type serious_std --n_samples 100`
2. `python -m hmmscan.scripts.validation.generate-samples --validation_type use_case --init_file_name scans/use_case/best_initializations/by_date_ex_iqr.csv --sequence_name dfb_by_date_ex_iqr_outliers --ae_type serious_std --n_samples 100`
3. `python -m hmmscan.scripts.validation.generate-samples --validation_type use_case --init_file_name scans/use_case/best_initializations/by_date_ex_iqr.csv --sequence_name dfc_by_date_ex_iqr_outliers --ae_type serious_std --n_samples 100`

## 2. Evaluate samples
Evaluate which model has the lowest BIC on each sample.
For this section, you will need to look at `ae-project/results/scans/use_case/best_initializations/[output_subdir].csv` and find the best number of states and mixture components for each `sequence_name` and `ae_type` combination.
The best structure is referred to below as `best_n_states` and `best_n_mix_comps`.
This best structure also defines a `grid_index`, where `grid_index = 9 * (best_n_states - 1) + best_n_mix_comps - 1`.

From the top level of this directory on Engaging, run the following command for each combination of `sequence_name` and `ae_type`:

`sbatch --array=0-99 hmmscan/cluster/evaluate-use-case-samples.sh scans/use_case/best_initializations/[output_subdir].csv [sequence_name] [ae_type] [best_n_states] [best_n_mix_comps] [grid_index]`.

This generates a file for each sequence name, ae type, candidate model, and sample id in `ae-project/validation/use_case/fits` and in `ae-project/validation/use_case/state_paths`.

### Paper Result Replication
Run the following three commands:

1. `sbatch --array=0-99 hmmscan/cluster/evaluate-use-case-samples.sh scans/use_case/best_initializations/by_date_ex_iqr.csv dfa_by_date_ex_iqr_outliers serious_std 3 2 19`
2. `sbatch --array=0-99 hmmscan/cluster/evaluate-use-case-samples.sh scans/use_case/best_initializations/by_date_ex_iqr.csv dfb_by_date_ex_iqr_outliers serious_std 3 3 20`
3. `sbatch --array=0-99 hmmscan/cluster/evaluate-use-case-samples.sh scans/use_case/best_initializations/by_date_ex_iqr.csv dfc_by_date_ex_iqr_outliers serious_std 2 3 11`

## 3. Aggregate the outputs
This script aggregates the outputs in `ae-project/validation/use_case/fits` and `ae-project/validation/use_case/state_paths`.

On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `python -m hmmscan.scripts.validation.aggregate_use_case_fits`.

These scripts create a CSV file `ae-project/validation/use_case/fits/all_fits.csv` aggregating the individual files from step 2.

## 4. Create the use case confidence intervals
Fit HMMs to the use case samples. 
This section uses `best_n_states` and `best_n_mix_comps` from step 2.

On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `python -m hmmscan.scripts.validation.evaluate-samples --validation_type ci --init_file_path scans/use_case/best_initializations/[output_subdir].csv --generating_sequence_name [sequence_name] --generating_ae_type [ae_type] --generating_n_states [best_n_states] --generating_n_mix_comps [best_n_mix_comps] --sample_index_min 0 --sample_index_max 99`.

### Paper Result Replication
Run these three commands:
1. `python -m hmmscan.scripts.validation.evaluate-samples --validation_type ci --init_file_path scans/use_case/best_initializations/by_date_ex_iqr.csv --generating_sequence_name dfa_by_date_ex_iqr_outliers --generating_ae_type serious_std --generating_n_states 3 --generating_n_mix_comps 2 --sample_index_min 0 --sample_index_max 99`
2. `python -m hmmscan.scripts.validation.evaluate-samples --validation_type ci --init_file_path scans/use_case/best_initializations/by_date_ex_iqr.csv --generating_sequence_name dfb_by_date_ex_iqr_outliers --generating_ae_type serious_std --generating_n_states 3 --generating_n_mix_comps 3 --sample_index_min 0 --sample_index_max 99`
3. `python -m hmmscan.scripts.validation.evaluate-samples --validation_type ci --init_file_path scans/use_case/best_initializations/by_date_ex_iqr.csv --generating_sequence_name dfc_by_date_ex_iqr_outliers --generating_ae_type serious_std --generating_n_states 2 --generating_n_mix_comps 3 --sample_index_min 0 --sample_index_max 99`

## 5. Visualize the results
It is probably easiest to generate the necessary plots locally off Engaging. To do so, copy `ae-project/validation/use_case/fits/all_fits.csv` to your local version of `ae-project`.

Then, you can use the script `use_case_validation.R` in the `hmmscan/scipts/viz` directory to look at the model validation results.

For the confidence intervals, you can use the script `ci.R`.
  