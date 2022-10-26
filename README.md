# HMMScan: Surveillance of Adverse Event Variability across Manufacturing Lots in Biologics

This repository implements the HMMScan method described in "Surveillance of Adverse Event Variability across Manufacturing Lots in Biologics".

## Installation
All file paths in this section are relative to the top level of this directory.

### Install Python packages into a virtual environment
1. Install Python 3.9 and make sure this python version is active.
2. Clone this repo and navigate to it locally.
3. Create a virtual environment named `venv`: `python -m venv venv`. It will be ignored by git.
4. Activate this virtual environment: `source venv\bin\activate`.
5. Install the required python packages from `requirements.txt`: `python -m pip install -r requirements.txt`.

### Install R packages into a virtual environment
1. Install `R` version 4.1. Make sure that this version is active (when called by `R` from command line) if you have multiple R versions.
2. Run R from the command line: `R`.
3. Restore the virtual R environment by running the following command in `R`: `renv::restore()`.

**Warning**: this last step may take a long time, but you only need to run it once.

### Download data
1. Download the Mendeley Data repository locally. # TODO: need to put the link to the repository in here 
2. Create `shared-path.txt` (in the top-level of this directory) that contains only the absolute local path of the Mendeley Data repository (e.g., `/Users/username/ae-project`). Do not include a new line character after the path.

## Replicating Paper Results

The Mendeley Data directory contains intermediate numerical results required to recreate all figures in the paper.
To replicate the paper figures only, please refer to the Paper Figures section below.

To recreate the intermediate results starting from the raw input data, follow the steps in the associated readme files referenced below.
The file paths referenced below assumes the Mendeley Data directory is downloaded to a directory called `ae-project`.

### Use Case HMMScan: BIC results, State Prediction
Refer to [Use Case Scanner Documentation](docs/use-case-scanner.md).

### Method Validation: Simulation
Refer to [Simulation Validation Documentation](docs/simulation-validation.md).
 
### Method Validation: Use Case
Refer to [Use Case Validation Documentation](docs/use-case-validation.md).

### Confidence Intervals
Refer to [Use Case Validation Documentation](docs/use-case-validation.md).

## New Use Case
Refer to the [New Use Case Documentation](docs/new-use-case-data.md) to ensure that the raw data is formatted correctly.
Then, refer to the documentation files referenced above to execute HMMScan.

### Paper Figures
This section provides references for the scripts and files that are used to generate the figures in the paper and supplementary material.
All scripts are found in `hmmscan/scripts/viz`.

#### Figures
- Figure 2: `s2c1-sim-results.R`
- Figure 3: `bic.R`
- Figure 4: `best_model_dists_and_predictions.R`
- Figure 5: `best_model_dists_and_predictions.R`
- Figure 6: `use_case_validation.R`

#### Tables
- Table 3: `summary_stats.R`
- Table 4: `best_model_dists_and_predictions.R` and `ci.R`