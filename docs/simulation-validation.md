# Simulation Validation Documentation
This file contains instructions for creating the Model Validation results.

## Required Inputs
1. `ae-project/validation/manual_input.csv`: this file contains manual specifications for the simulation scenarios.

## 1. Expand the manual simulation input
This script calculates the full set of input parameters necessary for each simulation scenario (also called an *experiment*) based on the manual input.

On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `python -m hmmscan.scripts.validation.expand-manual-sim-input manual_input.csv full_simulation_input.csv`

This generates `ae-project/validation/full_simulation_input.csv`.

## 2. Generate samples
There are 3030 experiment ids (with id numbers 1 through 3030) to run for the simulations.
This command runs a single experiment id at a time, and is designed to be used with an array on Engaging.
The array element number signifies the experiment id.

Since there are so many experiments and Engaging has a 500 job limit per user, you will need to break up the `sbatch` call into 7 increments. 
Here is the first call to make, subsequent calls should adjust the array indices until all 3030 experiments are run.

From the top level of this repository, run `sbatch --array=1-500 hmmscan/cluster/generate-sim-samples.sh`.

This script generates a file for each experiment id in `ae-project/validation/simulation/samples` and `ae-project/validation/simulation/state_paths`.

## 3. Evaluate samples
Since there are so many experiments and Engaging has a 500 job limit per user, you will need to break up the `sbatch` call into 7 increments.
Here is the first call to make, subsequent calls should adjust the array indices until all 3030 experiments are run.

From the top level of this repository, run `sbatch --array=1-500 hmmscan/cluster/evaluate-sim-samples.sh`.

This generates a file for each experiment id in `ae-project/validation/simulation/fits`.

## 4. Aggregate the outputs
This script aggregates the outputs in `ae-project/validation/simulation/fits` and `ae-project/validation/simulation/state_paths`.

On Engaging, run the following commands in an interactive session:

1. Load `python 3.9`: `module load python/3.9.4`.
2. Run `python -m hmmscan.scripts.validation.aggregate_sim_fits`.
3. Run `python -m hmmscan.scripts.validation.aggregate_sim_true_paths`.

These scripts create a CSV file for each model structure in the `state_paths` and `fits` directories aggregating the individual files from step 3.

## 5. Visualize the results
It is probably easiest to generate the necessary plots locally off Engaging. To do so, copy everything except the `samples` subdirectory from the `ae-project/validation/simulation/` directory to your local version of `ae-project`.

Then, you can use the scripts that end with `sim-results.R` in the `hmmscan/scipts/viz` directory.